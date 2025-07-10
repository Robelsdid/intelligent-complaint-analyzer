import gradio as gr
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import sys
import os

# Add src to path
sys.path.append('.')
from src.rag_utils import retrieve_relevant_chunks, build_prompt, generate_answer

class ChatInterface:
    def __init__(self):
        """Initialize the RAG pipeline components."""
        self.embed_model = None
        self.faiss_index = None
        self.metadata_df = None
        self.llm_pipeline = None
        self.is_loaded = False
    
    def load_models(self):
        """Load all models and data."""
        try:
            print("Loading embedding model...")
            self.embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            print("Loading FAISS index...")
            self.faiss_index = faiss.read_index('vector_store/complaints_faiss.index')
            
            print("Loading metadata...")
            self.metadata_df = pd.read_csv('data/chunked_complaints.csv')
            
            print("Loading LLM pipeline...")
            self.llm_pipeline = pipeline(
                "text2text-generation",
                model="google/flan-t5-small",
                max_length=200,
                min_length=30,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                device=0  # Use GPU if available
            )
            
            self.is_loaded = True
            print("All models loaded successfully!")
            return " Models loaded successfully! Ready to answer questions."
        except Exception as e:
            print(f"Error loading models: {e}")
            return f" Error loading models: {str(e)}"
    
    def process_question(self, question, history):
        """Process a question and return answer with sources."""
        if not self.is_loaded:
            return "Please load the models first.", "", []
        
        if not question.strip():
            return "Please enter a question.", "", []
        
        try:
            # Get answer and sources
            answer, sources = self._rag_answer(question)
            
            # Format sources for display
            formatted_sources = self._format_sources(sources)
            
            # Add to history (using messages format)
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})
            
            return formatted_sources, history
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            print(error_msg)
            return error_msg, history
    
    def _rag_answer(self, question, k=5):
        """Get RAG answer with sources."""
        # Retrieve relevant chunks
        chunks = retrieve_relevant_chunks(
            question, self.embed_model, self.faiss_index, self.metadata_df, k=k
        )
        
        # Add chunk_text if not present
        if 'chunk_text' not in chunks[0]:
            for i, c in enumerate(chunks):
                c['chunk_text'] = self.metadata_df.iloc[i]['chunk_text']
        
        # Build prompt and generate answer
        prompt = build_prompt(chunks, question)
        answer = generate_answer(prompt, self.llm_pipeline)
        
        return answer, chunks
    
    def _format_sources(self, sources):
        """Format source chunks for display."""
        formatted = []
        for i, source in enumerate(sources[:3], 1):  # Show top 3 sources
            text = source['chunk_text'][:300] + "..." if len(source['chunk_text']) > 300 else source['chunk_text']
            formatted.append(f"**Source {i}:** {text}")
        return "\n\n".join(formatted)
    
    def clear_history(self):
        """Clear the conversation history."""
        return [], ""

# Global instance
chat_interface = ChatInterface() 