import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from transformers import AutoTokenizer

class TabStarPreprocessor:
    """
    Préparateur de données éducatif pour nanoTabSTAR.
    
    Ce module condense la logique de preprocessing du papier TabSTAR original :
    1. Nettoyage des noms de colonnes (Text Cleaning)
    2. Expansion des dates (Date Expansion)
    3. Verbalisation des numériques avec information de Quantile (Feature Verbalization)
    4. Standardisation des numériques pour le MLP (Z-Score + Clipping)
    5. Création des tokens "Target-Aware" (Conscience de la cible)
    """

    def __init__(self, model_name='intfloat/e5-small-v2', max_token_len=32, n_quantiles=10):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_token_len = max_token_len
        self.n_quantiles = n_quantiles
        
        # Stockage des scalers et métadonnées
        self.num_scaler = None
        self.label_encoder = None
        self.numerical_cols = []
        self.text_cols = []
        self.target_name = None
        self.quantile_boundaries = {} # Stocke les bins pour chaque col numérique

    # ------------------------------------------------------------------
    # Pipeline utilisé par create_dump/build_h5_corpus (classification ou régression)
    # ------------------------------------------------------------------
    def process_dataset(self, df: pd.DataFrame, target_col: str, task_type: str = 'classification'):
        """Pipeline simple (verbalisation + num standardisé + tokens target-aware).

        - Densifie les colonnes sparse (OpenML en fournit parfois) pour éviter l'erreur
          pandas "cannot perform std with type Sparse".
        - Standardise les numériques (z-score, clip).
        - Verbalise chaque feature.
        - Tokenise features et cibles.
        """
        df = df.copy()
        df = self._clean_column_names(df)
        df = self._densify_sparse(df)

        features = [c for c in df.columns if c != target_col]

        # 1) Numériques
        num_df = df[features].select_dtypes(include=[np.number])
        verbalized_nums = pd.DataFrame(index=df.index)
        if not num_df.empty:
            means, stds = num_df.mean(), num_df.std()
            stds = stds.replace(0, 1e-6)
            z_scores = (num_df - means) / (stds + 1e-6)
            z_scores = z_scores.clip(-3, 3)
            processed_nums = z_scores.copy()
            for col in num_df.columns:
                q_str = self._get_quantile_str(num_df[col])
                verbalized_nums[col] = f"{col}: " + num_df[col].astype(str) + q_str
        else:
            processed_nums = pd.DataFrame(index=df.index)
            verbalized_nums = pd.DataFrame(index=df.index)

        # 2) Catégorielles / texte
        cat_df = df[features].select_dtypes(exclude=[np.number])
        verbalized_cats = pd.DataFrame(index=df.index)
        for col in cat_df.columns:
            verbalized_cats[col] = f"{col}: " + df[col].astype(str)
            processed_nums[col] = 0.0

        all_cols = list(verbalized_nums.columns) + list(verbalized_cats.columns)
        final_num_values = processed_nums[all_cols].fillna(0.0).values.astype(np.float32) if all_cols else np.zeros((len(df), 0), dtype=np.float32)
        feature_texts = pd.concat([verbalized_nums, verbalized_cats], axis=1)[all_cols].values if all_cols else np.empty((len(df), 0))

        # 3) Labels & target-aware tokens
        if task_type == 'regression':
            labels = df[target_col].astype(float).fillna(0).values
            target_strings = [f"Target {target_col}: value"]
        else:
            unique_classes = sorted(df[target_col].astype(str).unique())
            class_to_idx = {c: i for i, c in enumerate(unique_classes)}
            labels = df[target_col].astype(str).map(class_to_idx).values
            target_strings = [f"Target {target_col}: {c}" for c in unique_classes]

        flat_feature_texts = feature_texts.flatten().astype(str)
        if len(flat_feature_texts) == 0:
            feat_input_ids = np.zeros((len(df), 0, self.max_token_len), dtype=np.int64)
        else:
            feat_enc = self.tokenizer(
                flat_feature_texts.tolist(),
                padding='max_length',
                truncation=True,
                max_length=self.max_token_len,
                return_tensors='np'
            )
            n_samples, n_feats = feature_texts.shape
            feat_input_ids = feat_enc['input_ids'].reshape(n_samples, n_feats, self.max_token_len)

        target_enc = self.tokenizer(
            target_strings,
            padding='max_length',
            truncation=True,
            max_length=self.max_token_len,
            return_tensors='np'
        )

        return {
            "feature_input_ids": feat_input_ids.astype(np.int64),
            "feature_num_values": final_num_values.astype(np.float32),
            "target_input_ids": target_enc['input_ids'].astype(np.int64),
            "labels": labels.astype(np.int64) if task_type != 'regression' else labels.astype(np.float32),
            "n_classes": len(target_strings)
        }

    def _densify_sparse(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.apply(lambda s: s.sparse.to_dense() if pd.api.types.is_sparse(s) else s)

    def _get_quantile_str(self, series: pd.Series):
        try:
            labels = [f"{int(i*100/self.n_quantiles)}-{int((i+1)*100/self.n_quantiles)}%" for i in range(self.n_quantiles)]
            bins = pd.qcut(series, q=self.n_quantiles, labels=labels, duplicates='drop')
            return bins.astype(str).apply(lambda x: f" (Quantile {x})")
        except ValueError:
            return pd.Series([""] * len(series), index=series.index)

    def fit_transform(self, df: pd.DataFrame, target_col: str):
        """Apprend les statistiques (fit) et transforme le dataset (transform)."""
        print(f"--- Démarrage du Preprocessing TabSTAR sur {len(df)} lignes ---")
        self.target_name = target_col
        
        # 1. Nettoyage initial et séparation X/y
        df = self._clean_column_names(df)
        y = df[self.target_name]
        X = df.drop(columns=[self.target_name])
        
        # 2. Gestion des Dates (Inspiré de dates.py)
        # On explose les dates en (Année, Mois, Jour, etc.) AVANT la détection de type
        X = self._expand_date_features(X)

        # 3. Détection des Types (Inspiré de detection.py)
        self._detect_column_types(X)
        print(f"Colonnes détectées -> Numériques: {len(self.numerical_cols)}, Texte/Cat: {len(self.text_cols)}")

        # 4. Traitement de la Cible (Inspiré de target.py)
        # Pour la classification, on utilise un LabelEncoder
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y.astype(str))
        
        # 5. Apprentissage des Statistiques Numériques (Inspiré de scaler.py et binning.py)
        if self.numerical_cols:
            # A. Z-Score scaler pour la partie MLP (Mean=0, Std=1)
            self.num_scaler = StandardScaler()
            self.num_scaler.fit(X[self.numerical_cols].fillna(0)) # Simple fillna pour le fit
            
            # B. Calcul des Quantiles pour la partie Texte (Verbalisation)
            for col in self.numerical_cols:
                # On utilise qcut pour trouver les bornes des bins
                try:
                    # On ne garde que les valeurs valides pour calculer les quantiles
                    valid_values = X[col].dropna()
                    _, bins = pd.qcut(valid_values, self.n_quantiles, retbins=True, duplicates='drop')
                    self.quantile_boundaries[col] = bins
                except ValueError:
                    # Fallback si colonne constante ou trop petite
                    self.quantile_boundaries[col] = None

        # 6. Transformation effective (Application de la logique)
        return self._transform_data(X, y_encoded)

    def _transform_data(self, X: pd.DataFrame, y_encoded: np.ndarray):
        """Applique les transformations (Verbalisation & Standardisation)."""
        
        # --- PARTIE A : Numérique Dense (pour MLP) ---
        # Inspiré de scaler.py : Z-score avec clipping à +/- 3
        if self.numerical_cols:
            X_num = X[self.numerical_cols].fillna(X[self.numerical_cols].mean())
            X_num_scaled = self.num_scaler.transform(X_num)
            X_num_scaled = np.clip(X_num_scaled, -3, 3) # Clipping TabSTAR spécifique
        else:
            X_num_scaled = np.zeros((len(X), 0))

        # --- PARTIE B : Verbalisation (pour LLM) ---
        # C'est le cœur de TabSTAR : transformer tout en texte riche.
        verbalized_rows = []
        
        # On prépare un DataFrame de textes
        X_text = pd.DataFrame(index=X.index)

        # 1. Verbalisation des Numériques (Inspiré de binning.py)
        # Format: "NomCol: Valeur (Quantile X-Y%)"
        for col in self.numerical_cols:
            X_text[col] = X[col].apply(lambda val: self._verbalize_numeric(col, val))
            
        # 2. Verbalisation des Textes/Catégories
        # Format: "NomCol: Valeur"
        for col in self.text_cols:
            X_text[col] = X[col].apply(lambda val: f"{col}: {val}" if pd.notnull(val) else f"{col}: Unknown")

        # Re-ordonnancement pour aligner avec le tenseur numérique
        # Important : L'ordre des colonnes texte doit matcher l'ordre des colonnes dense
        all_cols = self.numerical_cols + self.text_cols
        
        # Padding du tenseur numérique pour les colonnes purement textuelles
        # (Les colonnes texte ont 0.0 dans la partie MLP)
        zeros_padding = np.zeros((len(X), len(self.text_cols)))
        if X_num_scaled.shape[1] > 0:
            final_num_values = np.hstack([X_num_scaled, zeros_padding])
        else:
            final_num_values = zeros_padding

        final_text_values = X_text[all_cols].values

        # --- PARTIE C : Tokenization ---
        print("Tokenisation en cours (cela peut prendre quelques secondes)...")
        # On aplatit tout pour tokeniser en batch (beaucoup plus rapide)
        flat_texts = final_text_values.flatten().astype(str).tolist()
        
        encoded = self.tokenizer(
            flat_texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_token_len,
            return_tensors='np'
        )
        
        n_samples = len(X)
        n_features = len(all_cols)
        
        input_ids = encoded['input_ids'].reshape(n_samples, n_features, self.max_token_len)
        
        # --- PARTIE D : Target-Aware Tokens ---
        # On prépare les strings pour chaque classe possible : "Target {col}: {classe}"
        classes = self.label_encoder.classes_
        target_strings = [f"Target {self.target_name}: {c}" for c in classes]
        
        target_encoded = self.tokenizer(
            target_strings,
            padding='max_length',
            truncation=True,
            max_length=self.max_token_len,
            return_tensors='np'
        )

        return {
            "feature_input_ids": input_ids.astype(np.int64),      # (N, M, L)
            "feature_num_values": final_num_values.astype(np.float32), # (N, M)
            "target_token_ids": target_encoded['input_ids'].astype(np.int64), # (C, L)
            "labels": y_encoded.astype(np.int64), # (N,)
            "n_classes": len(classes)
        }

    # ================= MÉTHODES UTILITAIRES (Détails d'implémentation) =================

    def _verbalize_numeric(self, col_name, value):
        """
        Logique de binning textuel inspirée de binning.py.
        Ex: 45 -> "Age: 45 (Quantile 40-50%)"
        """
        if pd.isna(value):
            return f"{col_name}: Unknown Value" # Comme dans nulls.py
        
        bins = self.quantile_boundaries.get(col_name)
        if bins is None:
            return f"{col_name}: {value}"
            
        # Trouver dans quel bin se trouve la valeur
        # np.digitize retourne l'index du bin (1-based)
        bin_idx = np.digitize([value], bins)[0]
        
        # Calcul du pourcentage pour l'affichage (10 bins -> pas de 10%)
        # Si bin_idx=1 (le premier), c'est 0-10%
        pct_step = 100 // self.n_quantiles
        lower_pct = max(0, (bin_idx - 1) * pct_step)
        upper_pct = min(100, bin_idx * pct_step)
        
        return f"{col_name}: {value} (Quantile {lower_pct}-{upper_pct}%)"

    def _clean_column_names(self, df):
        """Inspiré de texts.py : Normalise les noms (enlève \n, trim espaces)."""
        def clean(text):
            text = str(text)
            for c in ['\n', '\r', '\t']:
                text = text.replace(c, ' ')
            return text.strip()
        
        df.columns = [clean(c) for c in df.columns]
        # On applique aussi au nom de la target si besoin
        if self.target_name:
            self.target_name = clean(self.target_name)
        return df

    def _expand_date_features(self, df):
        """Inspiré de dates.py : Transforme datetime -> Year, Month, Day, Weekday."""
        # On itère sur une copie pour éviter de modifier l'original pendant l'itération
        new_df = df.copy()
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                # Extraction des composants
                new_df[f"{col}_year"] = df[col].dt.year
                new_df[f"{col}_month"] = df[col].dt.month
                new_df[f"{col}_day"] = df[col].dt.day
                new_df[f"{col}_weekday"] = df[col].dt.weekday
                # On supprime la colonne date originale car elle n'est pas digestible telle quelle
                new_df.drop(columns=[col], inplace=True)
        return new_df

    def _detect_column_types(self, df):
        """
        Inspiré de detection.py : Heuristiques simple pour séparer Numérique et Texte.
        """
        self.numerical_cols = []
        self.text_cols = []
        
        for col in df.columns:
            # Si c'est explicitement numérique
            if pd.api.types.is_numeric_dtype(df[col]):
                # Vérification supplémentaire : Est-ce un faux numérique (ex: ID categ) ?
                # TabSTAR detection.py utilise un seuil unique (MAX_NUMERIC_FOR_CATEGORICAL=50)
                n_unique = df[col].nunique()
                if n_unique < 10 and n_unique < len(df) * 0.05:
                    # Peu de valeurs uniques -> probablement catégoriel
                    self.text_cols.append(col)
                else:
                    self.numerical_cols.append(col)
            else:
                self.text_cols.append(col)