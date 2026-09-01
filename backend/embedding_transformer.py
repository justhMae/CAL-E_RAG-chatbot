from sklearn.base import BaseEstimator, TransformerMixin
from backend.embeddings import get_embedding_model

class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    """Scikit-learn-compatible wrapper around the shared MiniLM model.

    fit() is a no-op on X/y (the encoder is pretrained/frozen) -- it just
    makes sure the cached model is loaded so transform() can use it.
    """

    def fit(self, X, y=None):
        get_embedding_model()
        return self

    def transform(self, X):
        model = get_embedding_model()
        texts = X.tolist() if hasattr(X, "tolist") else list(X)
        return model.encode(
            texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False
        )
