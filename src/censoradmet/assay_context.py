"""Assay/document context features + description embeddings (plan §4).

The assay-aware model predicts mu = f(x, e) + b(a) and log s = g(x, e) + r(a),
where `e` is a per-measurement CONTEXT vector (assay protocol / provenance) and
b(a), r(a) are per-assay random effects (handled in heads.HeteroscedasticMLP).
This module builds `e` from the measurement-level metadata:

  * categorical: assay_type (A/B/F/...), source_name, standard_relation
  * numeric:    confidence_score, document_year (centred/scaled), log1p(...) counts
  * text:       a low-dimensional embedding of assay_description

LEAKAGE DISCIPLINE: every encoder (category vocab, numeric standardiser, text
vectoriser + SVD) is fit on the TRAIN rows only and then applied to test rows.
Unseen categories map to an all-zero indicator; unseen numeric NaNs map to the
train mean; the text embedding of an unseen description is produced by the
train-fitted vectoriser (no refit). ContextEncoder.fit(...) then .transform(...)
enforces this.

Description embedding: default is a TF-IDF (word 1-2grams) -> TruncatedSVD
pipeline -- fully reproducible, no external model download. An optional
sentence-transformer path (use_transformer=True) is available if the environment
has it; it is off by default so results are deterministic and portable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# NOTE: standard_relation is deliberately EXCLUDED. It encodes the observed
# censoring direction (=, <, >), which is derived from the measurement itself and
# would not be known when predicting a compound before the assay is run. Including
# it leaks the label for prospective prediction, so the assay-context vector uses
# only genuinely a-priori assay/provenance metadata.
CATEGORICAL = ["assay_type", "source_name"]
NUMERIC = ["confidence_score", "document_year"]


@dataclass
class ContextEncoder:
    text_dim: int = 32
    use_transformer: bool = False
    max_categories: int = 40
    _cat_vocab: dict = field(default_factory=dict)
    _num_mean: dict = field(default_factory=dict)
    _num_std: dict = field(default_factory=dict)
    _tfidf: object = None
    _svd: object = None
    _st_model: object = None
    _feature_names: list = field(default_factory=list)
    _fitted: bool = False

    # ------------------------------------------------------------------ #
    def fit(self, meta: pd.DataFrame, train_idx: np.ndarray):
        m = meta.iloc[train_idx]
        # categorical vocabularies (top-k by train frequency; rest -> OOV zero)
        self._cat_vocab = {}
        for c in CATEGORICAL:
            if c not in m:
                continue
            vc = m[c].astype(str).value_counts().head(self.max_categories)
            self._cat_vocab[c] = {v: i for i, v in enumerate(vc.index)}
        # numeric standardisation on train
        self._num_mean, self._num_std = {}, {}
        for c in NUMERIC:
            if c not in m:
                continue
            vals = pd.to_numeric(m[c], errors="coerce").to_numpy(dtype=float)
            mu = np.nanmean(vals) if np.isfinite(vals).any() else 0.0
            sd = np.nanstd(vals) if np.isfinite(vals).any() else 1.0
            self._num_mean[c] = float(mu)
            self._num_std[c] = float(sd if sd > 1e-6 else 1.0)
        # text embedding fit on train descriptions
        self._fit_text(m)
        self._fitted = True
        self._build_feature_names()
        return self

    def _fit_text(self, m: pd.DataFrame):
        if "assay_description" not in m:
            self._tfidf = None
            self._svd = None
            return
        texts = m["assay_description"].fillna("").astype(str).tolist()
        if self.use_transformer:
            try:
                from sentence_transformers import SentenceTransformer
                self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
                return
            except Exception:
                self.use_transformer = False
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=4000,
                                      stop_words="english")
        try:
            Xt = self._tfidf.fit_transform(texts)
        except ValueError:
            # empty vocabulary (all-empty descriptions)
            self._tfidf = None
            self._svd = None
            return
        k = int(min(self.text_dim, max(1, Xt.shape[1] - 1)))
        self._svd = TruncatedSVD(n_components=k, random_state=0)
        self._svd.fit(Xt)

    # ------------------------------------------------------------------ #
    def _cat_block(self, meta: pd.DataFrame) -> np.ndarray:
        blocks = []
        for c in CATEGORICAL:
            if c not in self._cat_vocab:
                continue
            vocab = self._cat_vocab[c]
            oh = np.zeros((len(meta), len(vocab)), dtype=np.float32)
            vals = meta[c].astype(str).to_numpy()
            for i, v in enumerate(vals):
                j = vocab.get(v)
                if j is not None:
                    oh[i, j] = 1.0
            blocks.append(oh)
        return np.concatenate(blocks, axis=1) if blocks else np.zeros((len(meta), 0), np.float32)

    def _num_block(self, meta: pd.DataFrame) -> np.ndarray:
        cols = []
        for c in NUMERIC:
            if c not in self._num_mean:
                continue
            vals = pd.to_numeric(meta[c], errors="coerce").to_numpy(dtype=float)
            vals = np.where(np.isfinite(vals), vals, self._num_mean[c])
            cols.append(((vals - self._num_mean[c]) / self._num_std[c]).astype(np.float32))
        return np.stack(cols, axis=1) if cols else np.zeros((len(meta), 0), np.float32)

    def _text_block(self, meta: pd.DataFrame) -> np.ndarray:
        if "assay_description" not in meta:
            return np.zeros((len(meta), 0), np.float32)
        texts = meta["assay_description"].fillna("").astype(str).tolist()
        if self.use_transformer and self._st_model is not None:
            emb = self._st_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            return emb.astype(np.float32)
        if self._tfidf is None or self._svd is None:
            return np.zeros((len(meta), 0), np.float32)
        Xt = self._tfidf.transform(texts)
        return self._svd.transform(Xt).astype(np.float32)

    def transform(self, meta: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("ContextEncoder must be fit before transform")
        return np.concatenate(
            [self._cat_block(meta), self._num_block(meta), self._text_block(meta)],
            axis=1,
        ).astype(np.float32)

    def _build_feature_names(self):
        names = []
        for c in CATEGORICAL:
            if c in self._cat_vocab:
                names += [f"{c}={v}" for v in self._cat_vocab[c]]
        for c in NUMERIC:
            if c in self._num_mean:
                names.append(f"num:{c}")
        tdim = 0
        if self.use_transformer and self._st_model is not None:
            tdim = int(self._st_model.get_sentence_embedding_dimension())
        elif self._svd is not None:
            tdim = int(self._svd.n_components)
        names += [f"desc_svd_{i}" for i in range(tdim)]
        self._feature_names = names

    @property
    def dim(self):
        return len(self._feature_names)


def build_context_features(ds, train_idx, text_dim: int = 32, use_transformer: bool = False):
    """Convenience: fit a ContextEncoder on train rows and return (X_ctx, encoder).
    X_ctx aligns row-for-row with ds.meta."""
    enc = ContextEncoder(text_dim=text_dim, use_transformer=use_transformer).fit(ds.meta, train_idx)
    return enc.transform(ds.meta), enc
