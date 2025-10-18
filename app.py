import os
import tempfile
import time
from typing import List, Tuple

# Third-party libraries
import gradio as gr
from unstructured.partition.auto import partition
from pydantic import BaseModel, Field

# LangChain components
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# --- Configuration and Initialization ---
# Ollama Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "llama3" # Use an LLM model you have pulled via 'ollama pull llama3'
EMBEDDING_MODEL = "nomic-embed-text" # Use an embedding model like 'nomic-embed-text' or 'all-minilm'

# RAG configuration
VECTOR_DB_DIR = "./chroma_db"

# Global state for the vector store
vector_store = None
llm_instance = None
retriever = None

# --- Pydantic Schema for Structured Output ---
# This defines the JSON structure we want the LLM to return when critiquing the resume
class ResumeCritique(BaseModel):
    """Structured output for resume critique and suggestions."""
    summary_of_strengths: str = Field(description="A concise summary of the resume's major strong points.")
    key_areas_for_improvement: List[str] = Field(description="A list of 3-5 specific, actionable points for improvement (e.g., 'Quantify experience,' 'Clarify career goal.').")
    suggested_career_path: str = Field(description="A suggestion for the most suitable career path or next job role based on the skills and experience provided.")


# --- RAG Pipeline Functions ---

def initialize_llm_and_chain(llm_name: str, embedding_name: str):
    """Initializes the Ollama models and the global RAG chain."""
    global llm_instance, retriever, vector_store
    
    print(f"Initializing LLM: {llm_name} and Embeddings: {embedding_name}...")
    
    try:
        # 1. Initialize Ollama LLM and Embeddings
        llm_instance = Ollama(model=llm_name, base_url=OLLAMA_BASE_URL)
        
        # NOTE: Ollama can serve both LLMs and Embedding models!
        embeddings_instance = OllamaEmbeddings(
            model=embedding_name, 
            base_url=OLLAMA_BASE_URL,
            # Ensure the model is running during embedding generation
            # This is important for stability
            model_kwargs={"keep_alive": -1} 
        )

        # 2. Load the Vector Store from the directory
        if os.path.exists(VECTOR_DB_DIR):
            vector_store = Chroma(
                persist_directory=VECTOR_DB_DIR, 
                embedding_function=embeddings_instance
            )
            # 3. Create a retriever instance (allows searching the vector store)
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            print("Initialization successful. Vector Store loaded.")
        else:
            vector_store = None
            retriever = None
            print("Vector Store directory not found. Please upload a file first.")
        
        return "System Ready! You can now ask questions about the loaded resume."

    except Exception as e:
        error_message = f"Initialization Error. Is Ollama running and models '{llm_name}' and '{embedding_name}' pulled? Error: {e}"
        print(error_message)
        return error_message

def ingest_resume(file_path: str):
    """
    Loads, chunks, embeds, and stores the resume document.
    Uses unstructured for robust resume parsing (PDF, DOCX, etc.).
    """
    global vector_store, retriever
    
    if not llm_instance:
        return "System not initialized. Please run initialize_llm_and_chain first."

    # Step 1: Load Document using Unstructured
    try:
        # Unstructured is great at handling various file types like PDF and DOCX
        elements = partition(filename=file_path)
        raw_text = "\n\n".join([str(el) for el in elements])
        
        # Create a single LangChain Document
        document = Document(page_content=raw_text, metadata={"source": os.path.basename(file_path)})
    except Exception as e:
        return f"Error loading file with Unstructured. Make sure the file is valid. Error: {e}"

    # Step 2: Split Document into Chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    docs = text_splitter.split_documents([document])

    # Step 3: Create Embeddings and Store in Chroma
    # Re-initialize embeddings instance to handle persistence and model loading
    embeddings_instance = OllamaEmbeddings(
        model=EMBEDDING_MODEL, 
        base_url=OLLAMA_BASE_URL,
        model_kwargs={"keep_alive": -1}
    )

    vector_store = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings_instance,
        persist_directory=VECTOR_DB_DIR # This saves the vector store to disk
    )
    vector_store.persist() # Explicitly persist the data
    
    # Update the global retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    return f"Successfully processed {len(docs)} chunks from the resume. Ask me anything about it now!"

def critique_resume():
    """Uses a structured prompt to ask the LLM for a professional critique."""
    if not retriever:
        return "Error: Resume not loaded. Please upload a file first.", []

    # Get the raw text of the entire document to ensure the LLM has full context for critique
    # We retrieve all stored chunks here.
    all_chunks = vector_store.get()['documents']
    full_context = "\n\n---\n\n".join(all_chunks)

    # 1. Define the System Prompt for the critique task
    CRITIQUE_PROMPT_TEMPLATE = """
    You are a world-class professional resume and career consultant.
    Your task is to analyze the provided resume text below, critique it constructively, and suggest improvements.
    You MUST respond with a valid JSON object matching the requested schema.

    --- RESUME CONTEXT ---
    {context}
    --- END CONTEXT ---
    """
    
    critique_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CRITIQUE_PROMPT_TEMPLATE),
            ("human", "Analyze the resume and provide a summary of strengths, key areas for improvement, and a suggested career path. Respond ONLY with the JSON object.")
        ]
    )

    # 2. Create the chain for structured output
    # Ollama uses the format instruction to trigger structured response logic
    critique_chain = critique_prompt | llm_instance.with_structured_output(ResumeCritique)
    
    try:
        # 3. Invoke the chain
        print("Critique in progress...")
        critique_result = critique_chain.invoke({"context": full_context})
        
        # 4. Format the output
        markdown_output = f"""
### 🌟 Professional Resume Critique 🌟

**Summary of Strengths:**
{critique_result.summary_of_strengths}

**Key Areas for Improvement:**
{'\n'.join([f"- {item}" for item in critique_result.key_areas_for_improvement])}

**Suggested Career Path:**
{critique_result.suggested_career_path}
"""
        return markdown_output
    except Exception as e:
        return f"Error during critique generation. Ensure LLM model supports structured output (llama3 is recommended). Error: {e}", []


# --- Conversation Logic (RAG Chain) ---

def ask_question(question: str, history: List[Tuple[str, str]]):
    """Answers a user question using the RAG chain."""
    
    if not retriever:
        yield "Please upload and process a resume file first using the 'Upload Resume' tab."
        return

    # 1. Define the Prompt Template (Crucial for RAG)
    # The prompt tells the LLM to use the retrieved context.
    RAG_PROMPT_TEMPLATE = """
    You are a helpful and detailed resume assistant. Your goal is to answer questions based ONLY on the provided context (the user's resume). 
    If you cannot find the answer in the context, politely state that the information is not in the resume. 
    Do not use any external knowledge.
    
    --- CONTEXT ---
    {context}
    --- QUESTION ---
    {input}
    """
    
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    # 2. Create the Document Combination Chain
    # This chain takes the retrieved documents and stuffs them into the prompt.
    document_chain = create_stuff_documents_chain(llm_instance, prompt)
    
    # 3. Create the full RAG Retrieval Chain
    # This chain handles: (Question -> Retrieval) -> (Documents + Question -> LLM)
    rag_chain = create_retrieval_chain(retriever, document_chain)

    try:
        # 4. Invoke the chain
        response_stream = rag_chain.stream({"input": question})
        
        # 5. Stream the Response
        full_response = ""
        for chunk in response_stream:
            if 'answer' in chunk:
                # The 'answer' key is part of the final generated text output
                full_response += chunk['answer']
                yield full_response
                
    except Exception as e:
        yield f"An error occurred during generation. Error: {e}"


# --- Gradio Interface Setup ---

# Initializing state when the application starts
initialize_llm_and_chain(LLM_MODEL, EMBEDDING_MODEL)


with gr.Blocks(title="Local Ollama RAG Resume Assistant", theme=gr.themes.Soft()) as demo:
    gr.HTML(f"""
        <div style="text-align: center; max-width: 800px; margin: 0 auto; padding: 20px; background-color: #f7f9fb; border-radius: 12px; border: 1px solid #e0e7ff;">
            <h1 style="font-size: 2.2em; color: #1e3a8a; font-weight: 700;">Local RAG Resume Assistant</h1>
            <p style="color: #4b5563; font-size: 1.1em;">
                Powered by LangChain, Ollama ({LLM_MODEL} & {EMBEDDING_MODEL}), and ChromaDB.
                Upload your resume (PDF/DOCX) below, and then ask questions or request a professional critique.
            </p>
        </div>
    """)

    with gr.Tabs():
        
        # 1. UPLOAD AND INDEX TAB
        with gr.TabItem("📄 Upload & Index Resume"):
            gr.Markdown("### Step 1: Upload Your Resume")
            
            # File Upload Component
            file_upload = gr.File(
                label="Upload Resume (PDF, DOCX, TXT recommended)",
                file_types=[".pdf", ".docx", ".txt"]
            )
            
            # Button to trigger ingestion
            process_button = gr.Button("Process and Index Resume", variant="primary")
            
            # Output message for ingestion status
            ingestion_status = gr.Textbox(
                label="Processing Status", 
                interactive=False, 
                value="Please upload a file and click 'Process and Index Resume'."
            )
            
            # Event Listener for File Processing
            process_button.click(
                fn=ingest_resume,
                inputs=[file_upload],
                outputs=[ingestion_status]
            )
            
        # 2. Q&A CHAT TAB
        with gr.TabItem("💬 Ask Q&A"):
            gr.Markdown("### Step 2: Ask Questions About Your Resume")
            
            # Chat Interface
            chatbot = gr.ChatInterface(
                fn=ask_question,
                textbox=gr.Textbox(
                    placeholder="E.g., What are my three main skills?",
                    container=False
                ),
                title=f"Chat with Resume Data (LLM: {LLM_MODEL})"
            )

        # 3. CRITIQUE TAB
        with gr.TabItem("✨ Get Critique"):
            gr.Markdown("### Step 3: Get a Structured Critique and Suggestions")
            
            critique_button = gr.Button("Generate Professional Critique", variant="secondary")
            
            critique_output = gr.Markdown("Critique will appear here after clicking the button. Ensure a resume has been successfully processed first.")
            
            # Event Listener for Critique Generation
            critique_button.click(
                fn=critique_resume,
                inputs=None,
                outputs=[critique_output]
            )

if __name__ == "__main__":
    # Launch the Gradio application
    demo.launch()
