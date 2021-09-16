import numpy as np

from sentence_transformers import SentenceTransformer

from ..datasets.itext_embedding_model import ITextEmbeddingModel


# Sentence Transformer model from: https://huggingface.co/sentence-transformers/distilbert-base-nli-stsb-mean-tokens/tree/main
class SentenceTransformerModel(ITextEmbeddingModel):
    def __init__(self):
        self._model = SentenceTransformer("distilbert-base-nli-stsb-mean-tokens")
        self._emebdding_dim = self._model.get_sentence_embedding_dimension()

    def encode(self, x: np.ndarray):
        batch_size = x.shape[0]
        flattened_y = self._model.encode(x.flatten())
        return flattened_y.reshape(batch_size, -1)

    def decode(self, y: np.ndarray):
        return np.full((y.shape[0], 1), fill_value="NotImplemented: Decoding is not available")
