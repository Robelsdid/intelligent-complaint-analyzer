import numpy as np
import pandas as pd
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

def retrieve_relevant_chunks(
    question: str,
    embed_model: SentenceTransformer,
    faiss_index: faiss.Index,
    metadata_df: pd.DataFrame,
    k: int = 5
) -> List[dict]:
    """
    Embed the question, retrieve top-k similar chunks from FAISS, return list of dicts with chunk_text and metadata.
    """
    q_emb = embed_model.encode([question])
    D, I = faiss_index.search(np.array(q_emb).astype('float32'), k)
    results = []
    for idx in I[0]:
        row = metadata_df.iloc[idx].to_dict()
        results.append(row)
    return results

def build_prompt(context_chunks: List[dict], question: str) -> str:
    """
    Build the prompt for the LLM using retrieved context and the user question.
    """
    context = "\n---\n".join([chunk['chunk_text'] for chunk in context_chunks])
    prompt = (
        "You are a financial analyst assistant for CrediTrust. "
        "Analyze the following customer complaints and provide a comprehensive answer.\n\n"
        "REQUIREMENTS:\n"
        "1. Identify the main issues and problems\n"
        "2. Provide specific examples from the complaints\n"
        "3. Mention any patterns or recurring themes\n"
        "4. Give a detailed summary (3-4 sentences minimum)\n"
        "5. If no relevant information exists, say 'No relevant complaints found'\n\n"
        f"CUSTOMER COMPLAINTS:\n{context}\n\n"
        f"ANALYSIS REQUEST: {question}\n\n"
        "DETAILED RESPONSE:"
    )
    return prompt

def generate_answer(prompt: str, llm_pipeline) -> str:
    """
    Generate an answer using a Hugging Face text-generation pipeline.
    """
    result = llm_pipeline(prompt, max_new_tokens=256, do_sample=True)
    if isinstance(result, list):
        return result[0]['generated_text'] if 'generated_text' in result[0] else result[0]['text']
    return result
