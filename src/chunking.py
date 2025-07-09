from typing import List
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_narratives(
    df: pd.DataFrame,
    text_col: str = 'cleaned_narrative',
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    metadata_cols: List[str] = None
) -> pd.DataFrame:
    """
    Chunk narratives in a DataFrame using RecursiveCharacterTextSplitter.
    Returns a new DataFrame with columns: chunk_text, chunk_id, and metadata.
    """
    if metadata_cols is None:
        metadata_cols = ['Product', 'Complaint ID'] if 'Complaint ID' in df.columns else ['Product']

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    rows = []
    for idx, row in df.iterrows():
        text = row[text_col]
        if not isinstance(text, str) or not text.strip():
            continue
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            meta = {col: row[col] for col in metadata_cols if col in row}
            rows.append({
                'chunk_text': chunk,
                'chunk_id': f"{idx}_{i}",
                **meta
            })
    return pd.DataFrame(rows) 