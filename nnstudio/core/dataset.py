"""Dataset loading and adaptation.

Any tabular file (CSV/TSV/Excel/JSON/Parquet) or a built-in demo dataset is
turned into the float32 tensors a Keras model consumes. The fitted
preprocessing is kept so that a single raw row can be transformed later for
inference.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

TASK_BINARY = "binary"
TASK_MULTICLASS = "multiclass"
TASK_REGRESSION = "regression"

TASK_LABELS = {
    TASK_BINARY: "Binary classification",
    TASK_MULTICLASS: "Multiclass classification",
    TASK_REGRESSION: "Regression",
}

MAX_CATEGORY_CARDINALITY = 50


class DatasetError(RuntimeError):
    """Raised when a dataset cannot be loaded or adapted."""


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# --------------------------------------------------------------------- loading

READABLE_SUFFIXES = (".csv", ".txt", ".tsv", ".xlsx", ".xls", ".json", ".parquet")

FILE_FILTER = (
    "Tabular data (*.csv *.tsv *.txt *.xlsx *.xls *.json *.parquet);;All files (*)"
)


def load_file(path: str | Path) -> pd.DataFrame:
    """Read a tabular file into a DataFrame, sniffing the CSV delimiter."""
    p = Path(path)
    suffix = p.suffix.lower()
    try:
        if suffix in (".csv", ".txt"):
            frame = pd.read_csv(p, sep=None, engine="python")
        elif suffix == ".tsv":
            frame = pd.read_csv(p, sep="\t")
        elif suffix in (".xlsx", ".xls"):
            frame = pd.read_excel(p)
        elif suffix == ".json":
            frame = pd.read_json(p)
        elif suffix == ".parquet":
            frame = pd.read_parquet(p)
        else:
            raise DatasetError(f"Unsupported file type: {suffix or p.name}")
    except DatasetError:
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim in the UI
        raise DatasetError(f"Could not read {p.name}: {exc}") from exc

    if frame.empty:
        raise DatasetError(f"{p.name} contains no rows")
    frame.columns = [str(c) for c in frame.columns]
    return frame


BUILTIN_DATASETS = (
    "Iris - 4 features, 3 classes",
    "Wine - 13 features, 3 classes",
    "Breast cancer - 30 features, 2 classes",
    "Digits - 64 features, 10 classes",
    "Diabetes - 10 features, regression",
    "Two moons - 2 features, 2 classes",
    "Concentric circles - 2 features, 2 classes",
    "Positive sine - 1 feature, regression",
)


def load_builtin(label: str) -> tuple[pd.DataFrame, str]:
    """Return (dataframe, suggested target column) for a demo dataset."""
    from sklearn import datasets as skd

    key = label.split(" - ")[0]
    rng = np.random.default_rng(7)

    if key in ("Iris", "Wine", "Breast cancer", "Digits"):
        loader = {
            "Iris": skd.load_iris,
            "Wine": skd.load_wine,
            "Breast cancer": skd.load_breast_cancer,
            "Digits": skd.load_digits,
        }[key]
        data = loader(as_frame=True)
        frame = data.frame.copy()
        names = list(data.target_names)
        frame["target"] = [str(names[int(i)]) for i in data.target]
        return frame, "target"

    if key == "Diabetes":
        data = skd.load_diabetes(as_frame=True)
        return data.frame.copy(), "target"

    if key in ("Two moons", "Concentric circles"):
        maker = skd.make_moons if key == "Two moons" else skd.make_circles
        kwargs = {"noise": 0.2, "random_state": 7}
        if key == "Concentric circles":
            kwargs["factor"] = 0.5
        x, y = maker(n_samples=800, **kwargs)
        frame = pd.DataFrame(x, columns=["x1", "x2"])
        frame["target"] = ["class_a" if v == 0 else "class_b" for v in y]
        return frame, "target"

    if key == "Positive sine":
        x = rng.uniform(0.0, 4.0 * np.pi, size=700)
        y = 2.0 + np.sin(x) + rng.normal(0.0, 0.12, size=x.shape)
        return pd.DataFrame({"x": x, "target": y}), "target"

    raise DatasetError(f"Unknown built-in dataset: {label}")


# ------------------------------------------------------------------- inference

def infer_task(target: pd.Series) -> str:
    """Guess whether a target column is binary, multiclass or continuous."""
    clean = target.dropna()
    if clean.empty:
        raise DatasetError("The target column is empty")

    non_numeric = (
        clean.dtype == object
        or str(clean.dtype) in ("category", "bool", "string")
        or not pd.api.types.is_numeric_dtype(clean)
    )
    distinct = int(clean.nunique())

    if non_numeric:
        return TASK_BINARY if distinct == 2 else TASK_MULTICLASS
    if distinct == 2:
        return TASK_BINARY

    numeric = pd.to_numeric(clean, errors="coerce").dropna()
    integer_like = bool(np.allclose(numeric % 1, 0))
    if integer_like and distinct <= 20 and distinct < max(10, len(numeric) * 0.05):
        return TASK_MULTICLASS
    return TASK_REGRESSION


# ---------------------------------------------------------------------- bundle

@dataclass
class DataBundle:
    """Everything the model, the charts and the inference panel need."""

    name: str
    task: str
    target_name: str
    feature_names: list[str]
    class_names: list[str] | None
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    transformer: ColumnTransformer
    raw_features: pd.DataFrame
    raw_targets: np.ndarray
    val_index: np.ndarray
    numeric_columns: list[str]
    categorical_columns: list[str]
    raw_defaults: dict
    raw_choices: dict
    dropped_columns: list[str] = field(default_factory=list)
    target_mean: float = 0.0
    target_std: float = 1.0
    target_scaled: bool = False

    @property
    def n_inputs(self) -> int:
        return int(self.x_train.shape[1])

    @property
    def n_outputs(self) -> int:
        return int(self.y_train.shape[1])

    @property
    def n_train(self) -> int:
        return int(self.x_train.shape[0])

    @property
    def n_val(self) -> int:
        return int(self.x_val.shape[0])

    @property
    def raw_columns(self) -> list[str]:
        return list(self.raw_features.columns)

    def transform_row(self, values: dict) -> np.ndarray:
        """Turn one raw feature dict into a (1, n_inputs) model input."""
        row = {c: values.get(c, self.raw_defaults.get(c)) for c in self.raw_columns}
        frame = pd.DataFrame([row], columns=self.raw_columns)
        for column in self.numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        for column in self.categorical_columns:
            frame[column] = frame[column].astype(str)
        return np.asarray(self.transformer.transform(frame), dtype="float32")

    def sample_raw_row(self, rng=None) -> tuple:
        """Pick a random validation row: (raw feature values, true target text)."""
        if self.val_index.size == 0:
            raise DatasetError("There is no validation split to sample from")
        rng = rng or np.random.default_rng()
        position = int(rng.integers(0, self.val_index.size))
        index = int(self.val_index[position])
        values = self.raw_features.iloc[index].to_dict()
        truth = self.raw_targets[index]
        if self.task == TASK_REGRESSION:
            return values, f"{float(truth):.4f}"
        return values, str(truth)

    def describe_prediction(self, prediction) -> str:
        """Human-readable reading of a single raw model output."""
        vector = np.asarray(prediction).reshape(-1)
        if self.task == TASK_REGRESSION:
            value = float(vector[0])
            if self.target_scaled:
                value = value * self.target_std + self.target_mean
            return f"{self.target_name} = {value:.4f}"
        if self.task == TASK_BINARY:
            names = self.class_names or ["0", "1"]
            p = float(vector[0])
            winner = names[1] if p >= 0.5 else names[0]
            confidence = p if p >= 0.5 else 1.0 - p
            return f"{winner}   ({confidence * 100:.1f}% confidence, raw output {p:.4f})"
        names = self.class_names or [str(i) for i in range(vector.size)]
        order = np.argsort(vector)[::-1][:3]
        parts = [f"{names[i]} {vector[i] * 100:.1f}%" for i in order]
        return f"{names[int(order[0])]}   ({' | '.join(parts)})"



# --------------------------------------------------------------------- prepare

@dataclass
class PreparationSpec:
    """How to turn one raw frame into tensors. Reusable across folds."""

    frame: pd.DataFrame
    target_column: str
    name: str = "dataset"
    task: str | None = None
    val_split: float = 0.2
    scale_features: bool = True
    scale_target: bool = False
    seed: int = 42

    def prepare(self) -> DataBundle:
        """One holdout split."""
        prep = _preprocess(self)
        split = float(min(max(self.val_split, 0.05), 0.5))
        indices = np.arange(len(prep.raw_features))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=split,
            random_state=self.seed,
            stratify=_stratify_labels(prep, split),
        )
        return _assemble(self, prep, train_idx, val_idx, self.name)

    def folds(self, k: int) -> FoldSet:
        """k splits where every row is validated exactly once."""
        prep = _preprocess(self)
        k = int(min(max(k, 2), 10))
        rows = len(prep.raw_features)
        if rows < k * 2:
            raise DatasetError(
                f"{rows} usable rows cannot be split into {k} folds - "
                "lower k or bring more data"
            )

        stratified = prep.codes is not None and int(np.bincount(prep.codes).min()) >= k
        if stratified:
            splitter = StratifiedKFold(k, shuffle=True, random_state=self.seed)
            pieces = splitter.split(np.zeros(rows), prep.codes)
        else:
            splitter = KFold(k, shuffle=True, random_state=self.seed)
            pieces = splitter.split(np.zeros(rows))

        bundles = [
            _assemble(self, prep, train_idx, val_idx, f"{self.name} fold {i}/{k}")
            for i, (train_idx, val_idx) in enumerate(pieces, start=1)
        ]
        return FoldSet(bundles=bundles, k=k, stratified=stratified, task=prep.task)


@dataclass
class FoldSet:
    """The k bundles of a cross-validation, plus how they were built."""

    bundles: list
    k: int
    stratified: bool
    task: str

    def __len__(self) -> int:
        return len(self.bundles)

    def __iter__(self):
        return iter(self.bundles)


@dataclass
class _Prepared:
    """Frame-wide work that does not depend on which rows are validation."""

    raw_features: pd.DataFrame
    raw_targets: pd.Series
    numeric: list
    categorical: list
    dropped: list
    task: str
    y: np.ndarray
    codes: np.ndarray | None
    class_names: list | None


def _preprocess(spec: PreparationSpec) -> _Prepared:
    """Clean rows and encode the target. No fitting happens here."""
    frame = spec.frame
    target_column = spec.target_column
    if target_column not in frame.columns:
        raise DatasetError(f"Column '{target_column}' is not in the dataset")

    work = frame.dropna(subset=[target_column]).reset_index(drop=True)
    if work.empty:
        raise DatasetError("Every row has a missing target value")
    if len(work) < 10:
        raise DatasetError("At least 10 usable rows are required")

    raw_targets = work[target_column]
    raw_features = work.drop(columns=[target_column]).copy()
    if raw_features.shape[1] == 0:
        raise DatasetError("The dataset needs at least one feature column")

    task = spec.task or infer_task(raw_targets)

    # Bools behave better as numbers, and runaway categoricals are dropped
    # instead of exploding into thousands of one-hot columns.
    for column in raw_features.columns:
        if pd.api.types.is_bool_dtype(raw_features[column]):
            raw_features[column] = raw_features[column].astype("float64")

    numeric = raw_features.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in raw_features.columns if c not in numeric]

    dropped = [
        c
        for c in categorical
        if raw_features[c].nunique(dropna=True) > MAX_CATEGORY_CARDINALITY
    ]
    if dropped:
        categorical = [c for c in categorical if c not in dropped]
        raw_features = raw_features.drop(columns=dropped)
    if not numeric and not categorical:
        raise DatasetError("No usable feature columns were left after adaptation")

    for column in categorical:
        raw_features[column] = raw_features[column].astype(str)

    codes = None
    class_names = None
    if task == TASK_REGRESSION:
        numeric_target = pd.to_numeric(raw_targets, errors="coerce")
        if numeric_target.isna().all():
            raise DatasetError(
                f"'{target_column}' is not numeric - pick a classification task instead"
            )
        numeric_target = numeric_target.fillna(numeric_target.median())
        y = numeric_target.to_numpy(dtype="float32").reshape(-1, 1)
    else:
        encoder = LabelEncoder()
        codes = encoder.fit_transform(raw_targets.astype(str).to_numpy())
        class_names = [str(c) for c in encoder.classes_]
        if len(class_names) < 2:
            raise DatasetError("The target column has a single class")
        if task == TASK_BINARY and len(class_names) != 2:
            raise DatasetError(
                f"'{target_column}' has {len(class_names)} classes - "
                "use a multiclass (softmax) output instead of a binary one"
            )
        if task == TASK_BINARY:
            y = codes.astype("float32").reshape(-1, 1)
        else:
            y = np.eye(len(class_names), dtype="float32")[codes]

    return _Prepared(
        raw_features=raw_features,
        raw_targets=raw_targets,
        numeric=numeric,
        categorical=categorical,
        dropped=dropped,
        task=task,
        y=y,
        codes=codes,
        class_names=class_names,
    )


def _stratify_labels(prep: _Prepared, split: float):
    """Class labels for a stratified split, or None when it is not viable."""
    if prep.codes is None:
        return None
    counts = np.bincount(prep.codes)
    if counts.min() < 2 or len(prep.codes) * split < len(counts):
        return None
    return prep.codes


def _build_transformer(prep: _Prepared, scale_features: bool) -> ColumnTransformer:
    steps: list = []
    if prep.numeric:
        numeric_steps = [("impute", SimpleImputer(strategy="median"))]
        if scale_features:
            numeric_steps.append(("scale", StandardScaler()))
        steps.append(("num", Pipeline(numeric_steps), prep.numeric))
    if prep.categorical:
        steps.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", _one_hot_encoder()),
                    ]
                ),
                prep.categorical,
            )
        )
    return ColumnTransformer(steps, verbose_feature_names_out=False)


def _assemble(
    spec: PreparationSpec,
    prep: _Prepared,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    name: str,
) -> DataBundle:
    """Fit the preprocessing on the training rows only, then build the bundle.

    Fitting on every row before splitting would leak validation statistics
    (the scaler's mean, the imputer's median) into training, which quietly
    flatters every score the app reports.
    """
    transformer = _build_transformer(prep, spec.scale_features)
    train_features = prep.raw_features.iloc[train_idx]
    val_features = prep.raw_features.iloc[val_idx]

    transformer.fit(train_features)
    x_train = np.asarray(transformer.transform(train_features), dtype="float32")
    x_val = np.asarray(transformer.transform(val_features), dtype="float32")
    feature_names = [str(n) for n in transformer.get_feature_names_out()]

    y_train = prep.y[train_idx]
    y_val = prep.y[val_idx]
    target_mean, target_std, target_scaled = 0.0, 1.0, False
    if prep.task == TASK_REGRESSION and spec.scale_target:
        target_mean = float(y_train.mean())
        target_std = float(y_train.std()) or 1.0
        y_train = (y_train - target_mean) / target_std
        y_val = (y_val - target_mean) / target_std
        target_scaled = True

    raw_defaults: dict = {}
    raw_choices: dict = {}
    for column in prep.numeric:
        median = pd.to_numeric(train_features[column], errors="coerce").median()
        raw_defaults[column] = float(median) if pd.notna(median) else 0.0
    for column in prep.categorical:
        raw_choices[column] = sorted(
            prep.raw_features[column].dropna().unique().tolist()
        )
        modes = train_features[column].mode()
        raw_defaults[column] = str(modes.iloc[0]) if not modes.empty else ""

    return DataBundle(
        name=name,
        task=prep.task,
        target_name=spec.target_column,
        feature_names=feature_names,
        class_names=prep.class_names,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        transformer=transformer,
        raw_features=prep.raw_features,
        raw_targets=prep.raw_targets.to_numpy(),
        val_index=np.asarray(val_idx),
        numeric_columns=prep.numeric,
        categorical_columns=prep.categorical,
        raw_defaults=raw_defaults,
        raw_choices=raw_choices,
        dropped_columns=prep.dropped,
        target_mean=target_mean,
        target_std=target_std,
        target_scaled=target_scaled,
    )


def prepare(
    frame: pd.DataFrame,
    target_column: str,
    *,
    name: str = "dataset",
    task: str | None = None,
    val_split: float = 0.2,
    scale_features: bool = True,
    scale_target: bool = False,
    seed: int = 42,
) -> DataBundle:
    """Adapt a raw DataFrame into train/validation tensors."""
    return PreparationSpec(
        frame=frame,
        target_column=target_column,
        name=name,
        task=task,
        val_split=val_split,
        scale_features=scale_features,
        scale_target=scale_target,
        seed=seed,
    ).prepare()
