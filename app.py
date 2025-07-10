import gradio as gr
from src.chat_interface import chat_interface

def create_interface():
    """Create the Gradio interface."""
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        max-width: 800px !important;
        margin: auto !important;
    }
    .chat-message {
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    """
    
    with gr.Blocks(css=css, title="CrediTrust Complaint Analyzer") as demo:
        gr.Markdown(
            """
            # CrediTrust Complaint Analyzer
            
            **Intelligent RAG-Powered Chatbot for Financial Services**
            
            Ask questions about customer complaints across Credit Cards, Personal Loans, BNPL, Savings Accounts, and Money Transfers.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=3):
                # Model loading section
                gr.Markdown("### 🔧 System Setup")
                load_button = gr.Button(" Load Models", variant="primary")
                load_status = gr.Textbox(label="Status", interactive=False)
                
                # Chat interface
                gr.Markdown("### 💬 Ask Questions")
                chatbot = gr.Chatbot(
                    label="Conversation History",
                    height=400,
                    show_label=True,
                    type="messages"
                )
                
                with gr.Row():
                    question_input = gr.Textbox(
                        label="Enter your question:",
                        placeholder="e.g., Why are people unhappy with Buy Now, Pay Later?",
                        lines=2
                    )
                    submit_btn = gr.Button(" Ask", variant="primary")
                
                clear_btn = gr.Button(" Clear Conversation", variant="secondary")
                
                # Sources display
                gr.Markdown("###  Sources")
                sources_display = gr.Markdown(
                    label="Source Chunks",
                    value="Sources will appear here after asking a question."
                )
        
        with gr.Column(scale=1):
            gr.Markdown("###  Example Questions")
            gr.Markdown("""
            - Why are people unhappy with Buy Now, Pay Later?
            - What are the most common complaints about credit cards?
            - Are there issues with money transfers?
            - What problems do customers report with savings accounts?
            - Are there any fraud-related complaints?
            """)
            
            gr.Markdown("### ℹ️ About")
            gr.Markdown("""
            This system uses:
            - **RAG (Retrieval-Augmented Generation)**
            - **FAISS vector search**
            - **Sentence transformers**
            - **FLAN-T5 language model**
            
            Built for CrediTrust Financial Services.
            """)
        
        # Event handlers
        load_button.click(
            fn=chat_interface.load_models,
            outputs=load_status
        )
        
        submit_btn.click(
            fn=chat_interface.process_question,
            inputs=[question_input, chatbot],
            outputs=[sources_display, chatbot]
        )
        
        question_input.submit(
            fn=chat_interface.process_question,
            inputs=[question_input, chatbot],
            outputs=[sources_display, chatbot]
        )
        
        clear_btn.click(
            fn=chat_interface.clear_history,
            outputs=[chatbot, sources_display]
        )
    
    return demo

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=None,  # Let Gradio find available port
        share=False,  
        show_error=True
    ) 