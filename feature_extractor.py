import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from pathlib import Path

# Optional Word2Vec support via gensim
try:
    from gensim.models import Word2Vec, KeyedVectors
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

class TFIDFExtractor:
    """
    Wraps sklearn's TfidfVectorizer with save/load helpers.

    Usage
    -----
    extractor = TFIDFExtractor()
    matrix = extractor.fit_transform(list_of_preprocessed_strings)
    score   = extractor.similarity(resume_vec, jd_vec)
    """

    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,          # apply log normalization
        )
        self._fitted = False

    def fit_transform(self, documents: list[str]):
        """Fit on corpus and transform. Returns sparse matrix."""
        matrix = self.vectorizer.fit_transform(documents)
        self._fitted = True
        return matrix

    def transform(self, documents: list[str]):
        """Transform new documents using already-fitted vectorizer."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform() first.")
        return self.vectorizer.transform(documents)

    def similarity(self, vec_a, vec_b) -> float:
        """Cosine similarity between two sparse/dense vectors."""
        return float(cosine_similarity(vec_a, vec_b)[0][0])

    def save(self, path: str = "models/tfidf_vectorizer.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, path)
        print(f"Vectorizer saved → {path}")

    def load(self, path: str = "models/tfidf_vectorizer.pkl"):
        self.vectorizer = joblib.load(path)
        self._fitted = True
        print(f"Vectorizer loaded ← {path}")

class BagOfWordsExtractor:
    """Simple CountVectorizer wrapper."""

    def __init__(self, max_features: int = 3000):
        self.vectorizer = CountVectorizer(max_features=max_features)

    def fit_transform(self, documents: list[str]):
        return self.vectorizer.fit_transform(documents)

    def transform(self, documents: list[str]):
        return self.vectorizer.transform(documents)

    def get_feature_names(self) -> list[str]:
        return self.vectorizer.get_feature_names_out().tolist()

class Word2VecExtractor:
    """
    Trains a Word2Vec model on the resume corpus and represents each
    document as the mean of its word vectors.

    Requires: pip install gensim
    """

    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 1):
        if not GENSIM_AVAILABLE:
            raise ImportError("gensim not installed. Run: pip install gensim")
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None

    def fit(self, tokenized_docs: list[list[str]]):
        """Train Word2Vec on a list of token lists."""
        self.model = Word2Vec(
            sentences=tokenized_docs,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=4,
        )

    def transform(self, tokenized_docs: list[list[str]]) -> np.ndarray:
        """Return mean-pooled embedding for each document."""
        if self.model is None:
            raise RuntimeError("Call fit() first.")
        embeddings = []
        for tokens in tokenized_docs:
            vecs = [
                self.model.wv[t] for t in tokens if t in self.model.wv
            ]
            if vecs:
                embeddings.append(np.mean(vecs, axis=0))
            else:
                embeddings.append(np.zeros(self.vector_size))
        return np.array(embeddings)

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        dot = np.dot(vec_a, vec_b)
        norm = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        return float(dot / norm) if norm > 0 else 0.0
