"""Sentence-transformer embedding for articles.

Encodes article titles (and optionally snippets) into dense vectors
for downstream HDBSCAN clustering.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from newscheck.config import ClusterConfig
from newscheck.log import get_logger
from newscheck.models import Article

logger = get_logger(__name__)


class ArticleEmbedder:
    """Lazily loads a sentence-transformer model and encodes article text.

    The model is downloaded on first use and cached by the
    ``sentence-transformers`` library.

    Usage::

        embedder = ArticleEmbedder(cluster_config)
        vectors = embedder.embed(articles)
    """

    def __init__(self, config: ClusterConfig) -> None:
        self._model_name = config.embedding_model
        self._model: Any = None  # lazy-loaded

    def _load_model(self) -> Any:
        """Load the sentence-transformer model (once)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("loading_embedding_model", model=self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, articles: list[Article]) -> NDArray[np.float32]:
        """Encode articles into a 2-D float32 array of shape (N, dim).

        Each article is represented by ``title + ' | ' + snippet[:200]``
        to give the model enough context without overwhelming it.

        Args:
            articles: Non-empty list of articles to encode.

        Returns:
            numpy array of shape ``(len(articles), embedding_dim)``.
        """
        if not articles:
            return np.empty((0, 0), dtype=np.float32)

        texts = [self._article_to_text(art) for art in articles]
        model = self._load_model()

        logger.info("embedding_articles", count=len(texts))
        vectors: NDArray[np.float32] = model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vectors.astype(np.float32)

    @staticmethod
    def _article_to_text(article: Article) -> str:
        """Build the text string sent to the encoder."""
        parts = [article.title]
        snippet = (article.snippet or "")[:200].strip()
        if snippet:
            parts.append(snippet)
        return " | ".join(parts)
