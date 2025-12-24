import re
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Set, Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.preprocessing import StandardScaler, QuantileTransformer, LabelEncoder
from skrub import DatetimeEncoder

# --- Constants from TabSTAR ---
MISSING_VALUE = "Unknown Value"
VERBALIZED_QUANTILE_BINS = 10
Z_MAX_ABS_VAL = 3

@dataclass
class TabSTARData:
    """Container for processed tabular data."""
    d_output: int
    x_txt: np.ndarray  # Verbalized features
    x_num: np.ndarray  # Normalized numerical values
    y: Optional[Series] = None

    def __len__(self) -> int:
        return len(self.x_txt)

class TabSTARPreprocessor:
    """
    Autonomous Preprocessor that replicates TabSTAR's logic.
    
    This class handles:
    1. Column name normalization.
    2. Date expansion (Year, Month, Day, Weekday, etc.).
    3. Feature type detection (Numerical vs Textual).
    4. Numerical scaling (Z-score clipped at 3).
    5. Semantic verbalization (Quantile-based descriptions for numbers).
    6. Target-aware token generation.
    """

    def __init__(self, is_cls: bool, verbose: bool = False):
        self.is_cls = is_cls
        self.verbose = verbose
        self.date_transformers: Dict[str, DatetimeEncoder] = {}
        self.numerical_transformers: Dict[str, StandardScaler] = {}
        self.semantic_transformers: Dict[str, QuantileTransformer] = {}
        self.target_transformer: Optional[Union[LabelEncoder, StandardScaler]] = None
        self.d_output: Optional[int] = None
        self.y_name: Optional[str] = None
        self.y_values: Optional[List[str]] = None
        self.constant_columns: List[str] = []

    def vprint(self, msg):
        if self.verbose:
            print(msg)

    # --- 1. Fitting Logic ---

    def fit(self, X: DataFrame, y: Series):
        """Fits all transformers to the training data."""
        x = X.copy()
        y = y.copy()
        
        # Basic cleaning
        x, y = self._densify_objects(x, y)
        
        # Date expansion
        self.date_transformers = self._fit_date_encoders(x)
        self.vprint(f"Detected {len(self.date_transformers)} date features.")
        x = self._transform_date_features(x, self.date_transformers)
        
        # Normalize names
        x, y = self._replace_column_names(x, y)
        
        # Detect types
        numerical_features = self._detect_numerical_features(x)
        self.vprint(f"Detected {len(numerical_features)} numerical features.")
        
        # Transform types
        x = self._transform_feature_types(x, numerical_features)
        
        # Target fitting
        self.target_transformer = self._fit_preprocess_y(y, self.is_cls)
        self.d_output = len(self.target_transformer.classes_) if self.is_cls else 1
        self.y_name = str(y.name)
        if self.is_cls:
            self.y_values = sorted(self.target_transformer.classes_)
            
        # Numerical transformers
        self.constant_columns = [col for col in x.columns if x[col].nunique() == 1]
        for col in numerical_features:
            if col in self.constant_columns:
                continue
            self.numerical_transformers[col] = self._fit_standard_scaler(x[col])
            self.semantic_transformers[col] = self._fit_numerical_bins(x[col])

    # --- 2. Transformation Logic ---

    def transform(self, x: DataFrame, y: Optional[Series] = None) -> TabSTARData:
        """Transforms data into verbalized text and normalized numbers."""
        x = x.copy()
        if y is not None:
            y = y.copy()
            
        x, y = self._densify_objects(x, y)
        x = self._transform_date_features(x, self.date_transformers)
        x, y = self._replace_column_names(x, y)
        
        num_cols = sorted(self.numerical_transformers)
        x = self._transform_feature_types(x, set(num_cols))
        
        # Target transformation
        y_processed = self._transform_target(y)
        
        # Verbalization
        x = self._verbalize_textual_features(x)
        x = x.drop(columns=self.constant_columns, errors='ignore')
        
        # Prepend target tokens (Target-Awareness)
        x = self._prepend_target_tokens(x, self.y_name, self.y_values)
        
        text_cols = [col for col in x.columns if col not in num_cols]
        x_txt_df = x[text_cols + num_cols].copy()
        x_num = np.zeros(shape=x.shape, dtype=np.float32)
        
        for col in num_cols:
            # Semantic verbalization (e.g., "10 to 20 (Quantile 10-20%)")
            x_txt_df[col] = self._transform_numerical_bins(x[col], self.semantic_transformers[col])
            
            # Numerical scaling (Z-score)
            idx = x_txt_df.columns.get_loc(col)
            s_num = self._transform_clipped_z_scores(x[col], self.numerical_transformers[col])
            x_num[:, idx] = s_num.to_numpy()
            
        return TabSTARData(
            d_output=self.d_output,
            x_txt=x_txt_df.to_numpy(),
            x_num=x_num,
            y=y_processed
        )

    # --- Internal Helper Methods (The "Vended" Logic) ---

    def _densify_objects(self, x, y):
        for col in x.columns:
            if hasattr(x[col].dtype, 'sparse'):
                x[col] = x[col].sparse.to_dense()
        if y is not None and hasattr(y.dtype, 'sparse'):
            y = y.sparse.to_dense()
        return x, y

    def _fit_date_encoders(self, x):
        encoders = {}
        for col in x.columns:
            if pd.api.types.is_datetime64_any_dtype(x[col]):
                enc = DatetimeEncoder(add_weekday=True, add_total_seconds=True)
                enc.fit(pd.to_datetime(x[col], errors='coerce').apply(lambda dt: dt.tz_localize(None) if getattr(dt, 'tzinfo', None) else dt))
                encoders[col] = enc
        return encoders

    def _transform_date_features(self, x, encoders):
        for col, enc in encoders.items():
            s = pd.to_datetime(x[col], errors='coerce').apply(lambda dt: dt.tz_localize(None) if getattr(dt, 'tzinfo', None) else dt)
            dt_df = enc.transform(s)
            dt_df.index = x.index
            x = x.drop(columns=[col]).join(dt_df)
        return x

    def _replace_column_names(self, x, y):
        def normalize(text):
            text = re.sub(r'[\x00-\x1F\x7F]', ' ', str(text))
            if ' ' in text:
                for c in ['_', '-', ".", ":"]: text = text.replace(c, ' ')
            return text
        
        old2new = {c: normalize(c) for c in x.columns}
        x = x.rename(columns=old2new)
        if y is not None:
            y.name = normalize(y.name)
        return x, y

    def _detect_numerical_features(self, x):
        num_feats = set()
        for col in x.columns:
            s = x[col].dropna()
            if len(s) == 0: continue
            if pd.api.types.is_numeric_dtype(s.dtype):
                num_feats.add(col)
            else:
                # Check if mostly numeric strings
                try:
                    pd.to_numeric(s, errors='raise')
                    if s.nunique() > 50: num_feats.add(col)
                except: pass
        return num_feats

    def _transform_feature_types(self, x, num_feats):
        for col in x.columns:
            if col in num_feats:
                x[col] = pd.to_numeric(x[col], errors='coerce').astype(float)
            else:
                x[col] = x[col].astype(str).replace('nan', MISSING_VALUE)
        return x

    def _fit_standard_scaler(self, s):
        scaler = StandardScaler()
        scaler.fit(s.dropna().values.reshape(-1, 1))
        return scaler

    def _transform_clipped_z_scores(self, s, scaler):
        s_val = scaler.transform(s.fillna(s.mean()).values.reshape(-1, 1)).flatten()
        s_val = np.clip(s_val, -Z_MAX_ABS_VAL, Z_MAX_ABS_VAL)
        return Series(s_val, index=s.index)

    def _fit_numerical_bins(self, s):
        scaler = QuantileTransformer(n_quantiles=min(1000, len(s.dropna())), random_state=0)
        scaler.fit(s.dropna().values.reshape(-1, 1))
        return scaler

    def _transform_numerical_bins(self, s, scaler):
        q_levels = np.linspace(0, 1, VERBALIZED_QUANTILE_BINS + 1)
        boundaries = scaler.inverse_transform(q_levels.reshape(-1, 1)).flatten()
        
        def format_f(n):
            r = round(n, 4)
            return str(int(r)) if r.is_integer() else f"{r:.4f}".rstrip("0").rstrip(".")

        b_str = [format_f(b) for b in boundaries]
        bins = [f"Lower than {b_str[0]} (Q 0%)"]
        for i in range(len(b_str)-1):
            bins.append(f"{b_str[i]} to {b_str[i+1]} (Q {i*10}-{(i+1)*10}%)")
        bins.append(f"Higher than {b_str[-1]} (Q 100%)")
        
        bin_idx = np.digitize(s.fillna(np.nan), boundaries)
        verbalized = [bins[i] if not pd.isna(s.iloc[j]) else MISSING_VALUE for j, i in enumerate(bin_idx)]
        return [f"Predictive Feature: {s.name}\nFeature Value: {v}" for v in verbalized]

    def _verbalize_textual_features(self, x):
        for col in x.columns:
            if x[col].dtype == object:
                x[col] = x[col].apply(lambda v: f"Predictive Feature: {col}\nFeature Value: {v}")
        return x

    def _prepend_target_tokens(self, x, y_name, y_values):
        if y_values:
            tokens = [f"Target Feature: {y_name}\nFeature Value: {v}" for v in y_values]
        else:
            tokens = [f"Numerical Target Feature: {y_name}"]
        
        target_df = DataFrame({f"TARGET_TOKEN_{i}": [t] * len(x) for i, t in enumerate(tokens)}, index=x.index)
        return pd.concat([target_df, x], axis=1)

    def _fit_preprocess_y(self, y, is_cls):
        if is_cls:
            le = LabelEncoder()
            le.fit(y)
            return le
        else:
            return self._fit_standard_scaler(y)

    def _transform_target(self, y):
        if y is None: return None
        if self.is_cls:
            return Series(self.target_transformer.transform(y), index=y.index, name=y.name)
        else:
            return self._transform_clipped_z_scores(y, self.target_transformer)
