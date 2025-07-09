from typing import List, Union
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

def embed_texts(
    texts: Union[List[str], 'pd.Series'],
    model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
    batch_size: int = 32
) -> np.ndarray:
    """
    Generate embeddings for a list or Series of texts using a sentence-transformers model.
    Returns a numpy array of shape (n_texts, embedding_dim).
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(list(texts), batch_size=batch_size, show_progress_bar=True)
    return np.array(embeddings)
