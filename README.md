# Intelligent Complaint Analysis for Financial Services

##  RAG-Powered Chatbot for Customer Feedback Analysis

A comprehensive Retrieval-Augmented Generation (RAG) system that transforms unstructured customer complaints into actionable business intelligence for CrediTrust Financial.

##  Project Overview

CrediTrust Financial serves over 500,000 users across East African markets with financial products including Credit Cards, Personal Loans, BNPL, Savings Accounts, and Money Transfers. This project addresses the challenge of processing thousands of customer complaints to identify emerging issues and trends.

###  Business Objectives
- **Reduce time** for Product Managers to identify complaint trends from days to minutes
- **Empower non-technical teams** to get answers without data analyst intervention  
- **Enable proactive problem resolution** based on real-time customer feedback

##  Technical Architecture

### Core Components
- **Vector Database**: FAISS for efficient similarity search
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
- **Language Model**: Google FLAN-T5-small for answer generation
- **Web Interface**: Gradio for user-friendly interaction
- **GPU Acceleration**: for real-time inference

### Data Pipeline
1. **EDA & Preprocessing**: Clean and filter CFPB complaint data
2. **Text Chunking**: Split narratives into searchable chunks
3. **Vector Embedding**: Generate embeddings for semantic search
4. **RAG Pipeline**: Retrieve relevant chunks and generate answers
5. **Interactive UI**: Web interface for natural language queries

##  Quick Start

### Prerequisites
- Python 3.12 (PyTorch compatibility)
- CUDA-enabled GPU (optional but recommended)
- 8GB+ RAM

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/intelligent-complaint-analyzer.git
cd intelligent-complaint-analyzer
```

2. **Create virtual environment**
```bash
python -m venv rag-env
source rag-env/Scripts/activate  # Windows
# OR
source rag-env/bin/activate       # Mac/Linux
```

3. **Install dependencies**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pandas langchain sentence-transformers faiss-cpu transformers gradio
```

4. **Download data**
- Place CFPB complaint dataset in `data/data.csv`
- Run EDA notebook: `notebooks/eda_preprocessing.ipynb`

5. **Build vector database**
- Run chunking notebook: `notebooks/chunking.ipynb`
- Run embedding notebook: `notebooks/embedding_indexing.ipynb`

6. **Launch the interface**
```bash
python app.py
```

7. **Access the application**
- Open browser to `http://localhost:7861`
- Click " Load Models"
- Start asking questions!

##  Project Structure

```
intelligent-complaint-analyzer/
├── data/
│   ├── data.csv                    # Raw CFPB complaint data
│   ├── filtered_complaints.csv     # Cleaned and filtered data
│   └── chunked_complaints.csv      # Chunked narratives
├── notebooks/
│   ├── eda_preprocessing.ipynb     # Task 1: EDA and preprocessing
│   ├── chunking.ipynb              # Task 2: Text chunking
│   ├── embedding_indexing.ipynb    # Task 2: Embedding and indexing
│   └── rag_pipeline_and_eval.ipynb # Task 3: RAG evaluation
├── src/
│   ├── chunking.py                 # Modular chunking functions
│   ├── embedding.py                 # Embedding utilities
│   ├── rag_utils.py                # RAG pipeline logic
│   └── chat_interface.py           # Gradio interface backend
├── vector_store/
│   ├── complaints_faiss.index      # FAISS vector database
│   └── complaints_metadata.csv     # Chunk metadata
├── app.py                          # Main Gradio application
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

##  Usage Examples

### Sample Questions
- "Why are people unhappy with Buy Now, Pay Later?"
- "What are the most common complaints about credit cards?"
- "Are there issues with money transfers?"
- "What problems do customers report with savings accounts?"
- "Are there any fraud-related complaints?"

### Expected Output
The system provides:
- **Detailed answers** based on relevant complaint chunks
- **Source transparency** showing which complaints were used
- **Evidence-backed insights** for business decision making

##  Technical Details

### RAG Pipeline
1. **Question Embedding**: Convert user question to vector
2. **Semantic Search**: Find top-k similar complaint chunks
3. **Prompt Construction**: Combine question with retrieved context
4. **Answer Generation**: Generate response using FLAN-T5
5. **Source Attribution**: Display relevant complaint excerpts

### Performance Metrics
- **Average Quality Score**: 4.0/5 across test questions
- **Response Time**: <5 seconds with GPU acceleration
- **Retrieval Accuracy**: Top-5 relevant chunks per query

### Model Specifications
- **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Language Model**: FLAN-T5-small (80M parameters)
- **Vector Database**: FAISS IndexFlatL2
- **Chunk Size**: 300 tokens with 50 token overlap

##  Business Impact

### Key Performance Indicators (KPIs)
1.  **Time Reduction**: Complaint trend identification reduced from days to minutes
2.  **Accessibility**: Non-technical teams can get insights independently
3.  **Proactivity**: Real-time analysis enables proactive problem resolution

### Use Cases
- **Product Managers**: Identify emerging issues across financial products
- **Customer Support**: Understand common complaint patterns
- **Compliance Teams**: Monitor fraud and regulatory concerns
- **Executives**: Get strategic insights from customer feedback

##  Development

### Running Tests
```bash
# Test RAG pipeline
python -c "from src.rag_utils import *; print('RAG pipeline working!')"

# Test chat interface
python app.py
```

### Adding New Features
1. **New Embedding Models**: Update `src/embedding.py`
2. **Different LLMs**: Modify `src/chat_interface.py`
3. **UI Enhancements**: Edit `app.py` and CSS
4. **Data Sources**: Add new complaint datasets to `data/`

##  Evaluation Results

| Question | Quality Score | Key Insights |
|----------|---------------|--------------|
| BNPL Issues | 5/5 | Late fees, credit score impacts |
| Credit Card Complaints | 4/5 | Customer service, dispute processes |
| Money Transfer Problems | 4/5 | Security concerns, fee disputes |
| Savings Account Issues | 4/5 | Account closures, rate changes |
| Fraud Complaints | 3/5 | False alerts, poor response times |

##  Troubleshooting

### Common Issues
1. **CUDA Errors**: Ensure PyTorch CUDA installation
2. **Memory Issues**: Reduce chunk size or use CPU
3. **Import Errors**: Check virtual environment activation
4. **Port Conflicts**: Change server_port in app.py

### Performance Optimization
- **GPU Usage**: Set `device=0` for CUDA acceleration
- **Batch Processing**: Increase batch_size for faster inference
- **Memory Management**: Clear cache between large queries

##  Future Enhancements

### Technical Roadmap
- [ ] Response streaming for better UX
- [ ] Confidence scores for answers
- [ ] Multi-language support
- [ ] Real-time complaint monitoring
- [ ] Advanced analytics dashboard

### Business Features
- [ ] Automated alert system
- [ ] CRM integration
- [ ] Trend analysis visualization
- [ ] Sentiment analysis
- [ ] Complaint categorization

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- **CrediTrust Financial** for the business context and requirements
- **Consumer Financial Protection Bureau** for the complaint dataset
- **Hugging Face** for the transformer models and libraries
- **Facebook Research** for FAISS vector search
- **Gradio** for the web interface framework


**Built with love for transforming customer feedback into actionable insights** 
