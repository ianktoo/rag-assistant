# Standard library imports
import os  # For file system operations
import tempfile  # For temporary file handling
import time  # For timing operations
from typing import List, Tuple  # For type hints

# Third-party libraries
import gradio as gr  # Web UI framework for creating the interface
from unstructured.partition.auto import partition  # For parsing various document formats (PDF, DOCX, etc.)

# LangChain components for RAG (Retrieval-Augmented Generation)
from langchain_community.document_loaders import TextLoader  # For loading text documents
from langchain_community.vectorstores import Chroma  # Vector database for storing embeddings
from langchain_community.embeddings import OllamaEmbeddings  # Embedding model via Ollama
from langchain_community.llms import Ollama  # Large Language Model via Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter  # For splitting documents into chunks
from langchain.chains import create_retrieval_chain  # For creating RAG chains
from langchain.chains.combine_documents import create_stuff_documents_chain  # For combining retrieved documents
from langchain_core.prompts import ChatPromptTemplate  # For creating prompt templates
from langchain_core.documents import Document  # Document class for LangChain

# --- Configuration and Initialization ---
# Ollama Configuration - These settings connect to your local Ollama instance
OLLAMA_BASE_URL = "http://localhost:11434"  # Default Ollama server URL
LLM_MODEL = "llama3"  # Main language model for generating responses (must be pulled via 'ollama pull llama3')
EMBEDDING_MODEL = "nomic-embed-text"  # Embedding model for converting text to vectors (must be pulled via 'ollama pull nomic-embed-text')

# RAG (Retrieval-Augmented Generation) configuration
VECTOR_DB_DIR = "./chroma_db"  # Directory where ChromaDB will store the vector embeddings

# Global state variables - these hold our initialized components
vector_store = None  # ChromaDB vector store instance
llm_instance = None  # Ollama LLM instance
retriever = None  # Retriever for searching the vector store

# --- JSON Schema for Structured Output ---
# The LLM will be prompted to return JSON in this format for resume critiques
# This ensures consistent, structured responses that we can parse and format nicely


# --- RAG Pipeline Functions ---

def initialize_llm_and_chain(llm_name: str, embedding_name: str):
    """
    Initializes the Ollama models and the global RAG chain.
    
    This function sets up the core components needed for the RAG system:
    - LLM for generating responses
    - Embedding model for converting text to vectors
    - Vector store for storing and retrieving document chunks
    - Retriever for searching the vector store
    """
    global llm_instance, retriever, vector_store  # Access global variables
    
    print(f"Initializing LLM: {llm_name} and Embeddings: {embedding_name}...")
    
    try:
        # 1. Initialize Ollama LLM - this will handle text generation
        llm_instance = Ollama(model=llm_name, base_url=OLLAMA_BASE_URL)
        
        # 2. Initialize Ollama Embeddings - this converts text to numerical vectors
        # NOTE: Ollama can serve both LLMs and Embedding models from the same server!
        embeddings_instance = OllamaEmbeddings(
            model=embedding_name, 
            base_url=OLLAMA_BASE_URL,
            # Keep the model alive during embedding generation for better performance
            model_kwargs={"keep_alive": -1} 
        )

        # 3. Load existing Vector Store if it exists, or prepare for new one
        if os.path.exists(VECTOR_DB_DIR):
            # Load existing ChromaDB vector store with the embedding function
            vector_store = Chroma(
                persist_directory=VECTOR_DB_DIR, 
                embedding_function=embeddings_instance
            )
            # Create a retriever that will return top 5 most similar chunks
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            print("Initialization successful. Vector Store loaded.")
        else:
            # No existing vector store - will be created when first document is uploaded
            vector_store = None
            retriever = None
            print("Vector Store directory not found. Please upload a file first.")
        
        return "System Ready! You can now ask questions about the loaded resume."

    except Exception as e:
        # Handle initialization errors (e.g., Ollama not running, models not pulled)
        error_message = f"Initialization Error. Is Ollama running and models '{llm_name}' and '{embedding_name}' pulled? Error: {e}"
        print(error_message)
        return error_message

def ingest_resume(file_path: str):
    """
    Loads, chunks, embeds, and stores the resume document in the vector database.
    
    This is the core function that processes uploaded resumes and makes them searchable.
    It handles multiple file formats (PDF, DOCX, TXT) and creates embeddings for semantic search.
    
    Process:
    1. Parse the document using Unstructured (handles PDF, DOCX, etc.)
    2. Split the text into manageable chunks
    3. Convert chunks to embeddings using Ollama
    4. Store embeddings in ChromaDB for fast retrieval
    """
    global vector_store, retriever  # Access global variables
    
    # Check if the system has been initialized
    if not llm_instance:
        return "System not initialized. Please run initialize_llm_and_chain first."

    # Step 1: Load and Parse Document using Unstructured
    try:
        # Unstructured is excellent at handling various file types (PDF, DOCX, TXT, etc.)
        # It extracts text while preserving structure and formatting
        elements = partition(filename=file_path)
        # Combine all extracted elements into a single text string
        raw_text = "\n\n".join([str(el) for el in elements])
        
        # Create a LangChain Document object with metadata
        document = Document(page_content=raw_text, metadata={"source": os.path.basename(file_path)})
    except Exception as e:
        return f"Error loading file with Unstructured. Make sure the file is valid. Error: {e}"

    # Step 2: Split Document into Chunks for Better Retrieval
    # Large documents need to be split into smaller chunks for effective RAG
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Each chunk will be ~1000 characters
        chunk_overlap=200,    # 200 characters overlap between chunks for context
        length_function=len   # Use character count for splitting
    )
    docs = text_splitter.split_documents([document])  # Split into multiple documents

    # Step 3: Create Embeddings and Store in ChromaDB
    # Re-initialize embeddings instance to ensure proper model loading
    embeddings_instance = OllamaEmbeddings(
        model=EMBEDDING_MODEL, 
        base_url=OLLAMA_BASE_URL,
        model_kwargs={"keep_alive": -1}  # Keep model loaded for better performance
    )

    # Create vector store from documents and save to disk
    vector_store = Chroma.from_documents(
        documents=docs,                    # The chunked documents
        embedding=embeddings_instance,    # Embedding function
        persist_directory=VECTOR_DB_DIR   # Save to disk for persistence
    )
    vector_store.persist()  # Explicitly save the data to disk
    
    # Update the global retriever for searching the new vector store
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # Return top 5 most similar chunks

    return f"Successfully processed {len(docs)} chunks from the resume. Ask me anything about it now!"

def critique_resume():
    """
    Generates a professional, structured critique of the uploaded resume.
    
    This function provides comprehensive feedback including:
    - Overall assessment of resume quality
    - Key strengths and highlights
    - Areas for improvement (if any)
    - Suggested career path
    - Market confidence level
    
    The critique is designed to be encouraging for well-crafted resumes
    and constructive for those needing improvement.
    """
    # Check if a resume has been processed
    if not retriever:
        return "Error: Resume not loaded. Please upload a file first."

    # Get the complete resume text for comprehensive analysis
    # We retrieve all stored chunks to ensure the LLM has full context
    all_chunks = vector_store.get()['documents']  # Get all document chunks from vector store
    full_context = "\n\n---\n\n".join(all_chunks)  # Combine chunks with separators

    # 1. Define the System Prompt for structured critique generation
    # This prompt instructs the LLM to act as a professional career consultant
    # and return structured JSON output for consistent formatting
    CRITIQUE_PROMPT_TEMPLATE = """
    You are a world-class professional resume and career consultant.
    Your task is to analyze the provided resume text below and provide a comprehensive assessment.
    
    IMPORTANT: If the resume is already excellent and well-crafted, focus on highlighting its strengths and positive aspects rather than finding faults. Only suggest improvements if there are genuine areas that could be enhanced.
    
    You MUST respond with a valid JSON object in exactly this format:
    {{
        "overall_assessment": "Brief overall assessment of the resume quality (e.g., 'Excellent', 'Strong', 'Good with room for improvement', 'Needs significant work')",
        "summary_of_strengths": "A detailed summary of the resume's major strong points and what makes it effective.",
        "key_highlights": [
            "First standout feature or achievement",
            "Second impressive element",
            "Third notable strength"
        ],
        "key_areas_for_improvement": [
            "Only include if there are genuine areas for improvement. If resume is excellent, use 'No major improvements needed' or 'Minor suggestions only'"
        ],
        "suggested_career_path": "A suggestion for the most suitable career path or next job role based on the skills and experience provided.",
        "confidence_level": "Your confidence in this candidate's marketability (e.g., 'Very High', 'High', 'Moderate', 'Needs Development')"
    }}

    --- RESUME CONTEXT ---
    {context}
    --- END CONTEXT ---
    """
    
    # 2. Create the prompt template with system and human messages
    critique_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CRITIQUE_PROMPT_TEMPLATE),  # System message with instructions
            ("human", "Analyze the resume and provide a professional critique. Respond ONLY with the JSON object in the exact format specified above.")  # Human message with task
        ]
    )

    # 3. Create the chain for critique generation
    # This chains the prompt template with the LLM for structured output
    critique_chain = critique_prompt | llm_instance
    
    try:
        # 4. Invoke the chain to generate the critique
        print("Critique in progress...")
        response = critique_chain.invoke({"context": full_context})  # Pass the resume context
        
        # 5. Parse the JSON response from the LLM
        import json
        import re
        
        # Extract JSON from the response (LLM might include extra text)
        # Use regex to find the JSON object within the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()  # Extract just the JSON part
            critique_data = json.loads(json_str)  # Parse JSON
        else:
            # Fallback: try to parse the entire response as JSON
            critique_data = json.loads(response)
        
        # 6. Format the output with enhanced positive feedback and visual styling
        overall_assessment = critique_data.get('overall_assessment', 'Not provided')
        confidence_level = critique_data.get('confidence_level', 'Not provided')
        
        # Determine visual styling based on assessment quality
        # This creates a more engaging and encouraging user experience
        if 'excellent' in overall_assessment.lower() or 'very high' in confidence_level.lower():
            header_emoji = "🏆"  # Trophy for excellent resumes
            tone_color = "#10b981"  # Green color for excellent
        elif 'strong' in overall_assessment.lower() or 'high' in confidence_level.lower():
            header_emoji = "⭐"  # Star for strong resumes
            tone_color = "#3b82f6"  # Blue color for strong
        else:
            header_emoji = "📋"  # Clipboard for others
            tone_color = "#6b7280"  # Gray color for others
        
        # 7. Smart improvements section - only show if there are actual improvements needed
        # This prevents forced criticism of well-crafted resumes
        improvements = critique_data.get('key_areas_for_improvement', [])
        improvements_text = ""
        
        # Check if there are genuine improvements needed (not just "no improvements needed")
        if improvements and not any('no major improvements' in item.lower() or 'minor suggestions' in item.lower() for item in improvements):
            # Show constructive improvement suggestions
            improvements_text = f"""
**🔧 Areas for Enhancement:**
{chr(10).join([f"- {item}" for item in improvements])}
"""
        else:
            # Show positive reinforcement for excellent resumes
            improvements_text = """
**🎉 Excellent Work!**
This resume is well-crafted and doesn't require major improvements. The candidate has done an outstanding job presenting their qualifications.
"""
        
        # 8. Create the final formatted markdown output
        # This creates a beautiful, structured critique with emojis and colors
        markdown_output = f"""
### {header_emoji} Professional Resume Assessment {header_emoji}

**📊 Overall Assessment:** <span style="color: {tone_color}; font-weight: bold;">{overall_assessment}</span>

**🎯 Market Confidence:** <span style="color: {tone_color}; font-weight: bold;">{confidence_level}</span>

**💪 Key Strengths:**
{critique_data.get('summary_of_strengths', 'Not provided')}

**✨ Standout Highlights:**
{chr(10).join([f"- {item}" for item in critique_data.get('key_highlights', [])])}
{improvements_text}
**🚀 Recommended Career Path:**
{critique_data.get('suggested_career_path', 'Not provided')}
"""
        return markdown_output
        
    except json.JSONDecodeError as e:
        # Handle JSON parsing errors gracefully
        # Sometimes the LLM doesn't return perfect JSON, so we show the raw response
        return f"""
### 📋 Professional Resume Assessment 📋

**⚠️ Note:** The LLM response could not be parsed as structured JSON. Here's the raw response:

{response}

**Technical Error:** {e}
"""
    except Exception as e:
        # Handle any other errors during critique generation
        return f"Error during critique generation. Error: {e}"


# --- Conversation Logic (RAG Chain) ---

def ask_question(question: str, history: List[Tuple[str, str]]):
    """
    Answers user questions about their resume using RAG (Retrieval-Augmented Generation).
    
    This function implements the core RAG pipeline:
    1. Takes the user's question
    2. Retrieves relevant chunks from the resume using semantic search
    3. Combines the retrieved context with the question
    4. Generates an answer using the LLM
    5. Streams the response back to the user
    
    The function ensures answers are based only on the uploaded resume content.
    """
    # Check if a resume has been processed
    if not retriever:
        yield "Please upload and process a resume file first using the 'Upload Resume' tab."
        return

    # 1. Define the RAG Prompt Template
    # This template instructs the LLM to answer based only on the provided context
    # and to be honest when information isn't available in the resume
    RAG_PROMPT_TEMPLATE = """
    You are a helpful and detailed resume assistant. Your goal is to answer questions based ONLY on the provided context (the user's resume). 
    If you cannot find the answer in the context, politely state that the information is not in the resume. 
    Do not use any external knowledge.
    
    --- CONTEXT ---
    {context}
    --- QUESTION ---
    {input}
    """
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    # 2. Create the Document Combination Chain
    # This chain takes the retrieved documents and "stuffs" them into the prompt
    # It handles the formatting of multiple document chunks
    document_chain = create_stuff_documents_chain(llm_instance, prompt)
    
    # 3. Create the full RAG Retrieval Chain
    # This is the complete RAG pipeline: Question -> Retrieval -> Document Combination -> LLM
    rag_chain = create_retrieval_chain(retriever, document_chain)

    try:
        # 4. Invoke the RAG chain with streaming for real-time response
        response_stream = rag_chain.stream({"input": question})
        
        # 5. Stream the response back to the user in real-time
        # This provides a better user experience as the answer appears progressively
        full_response = ""
        for chunk in response_stream:
            if 'answer' in chunk:
                # The 'answer' key contains the generated text
                full_response += chunk['answer']
                yield full_response  # Yield the accumulated response so far
                
    except Exception as e:
        # Handle any errors during the RAG process
        yield f"An error occurred during generation. Error: {e}"


# --- Gradio Interface Setup ---

# Initialize the system when the application starts
# This sets up the LLM, embeddings, and loads any existing vector store
initialize_llm_and_chain(LLM_MODEL, EMBEDDING_MODEL)


# Theme Configuration - Choose your preferred theme
# Available options: gr.themes.Soft(), gr.themes.Default(), gr.themes.Glass(), 
# gr.themes.Monochrome(), gr.themes.Base(), or create a custom theme

# Option 1: Built-in themes (uncomment one)
# theme = gr.themes.Soft()  # Current - soft, rounded design
# theme = gr.themes.Default()  # Clean, minimal design
# theme = gr.themes.Glass()  # Modern glass-morphism effect
# theme = gr.themes.Monochrome()  # Black and white theme
# theme = gr.themes.Base()  # Basic theme

# Option 2: Custom theme with Nunito font and spacious, simple design
# This creates a clean, professional interface with the Nunito font
# and generous spacing for better readability and user experience
theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.blue,  # Primary color for buttons and highlights
    secondary_hue=gr.themes.colors.gray,  # Secondary color for accents
    neutral_hue=gr.themes.colors.slate,  # Neutral color for text and backgrounds
    font=gr.themes.GoogleFont("Nunito"),  # Nunito font for clean, friendly look
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),  # Monospace font for code
    # Spacious and simple design settings
    spacing_size=gr.themes.sizes.spacing_lg,  # More spacing between elements
    radius_size=gr.themes.sizes.radius_lg,    # Larger border radius for softer look
    text_size=gr.themes.sizes.text_lg,        # Larger text for better readability
)

# Option 3: More advanced custom themes (examples)
# Uncomment any of these to try different visual styles

# Professional dark theme - sophisticated look with indigo/purple colors
# theme = gr.themes.Soft(
#     primary_hue=gr.themes.colors.indigo,
#     secondary_hue=gr.themes.colors.purple,
#     neutral_hue=gr.themes.colors.slate,
#     font=gr.themes.GoogleFont("Roboto"),
#     font_mono=gr.themes.GoogleFont("Fira Code"),
# )

# Warm, friendly theme - approachable look with orange/amber colors
# theme = gr.themes.Soft(
#     primary_hue=gr.themes.colors.orange,
#     secondary_hue=gr.themes.colors.amber,
#     neutral_hue=gr.themes.colors.stone,
#     font=gr.themes.GoogleFont("Poppins"),
# )

# Modern tech theme - clean tech look with emerald/teal colors
# theme = gr.themes.Soft(
#     primary_hue=gr.themes.colors.emerald,
#     secondary_hue=gr.themes.colors.teal,
#     neutral_hue=gr.themes.colors.gray,
#     font=gr.themes.GoogleFont("Inter"),
# )

# Option 4: Fully custom theme with complete control
# This shows additional customization options for fine-tuning the appearance
# theme = gr.themes.Soft(
#     primary_hue=gr.themes.colors.blue,
#     secondary_hue=gr.themes.colors.gray,
#     neutral_hue=gr.themes.colors.slate,
#     font=gr.themes.GoogleFont("Inter"),
#     font_mono=gr.themes.GoogleFont("JetBrains Mono"),
#     # Additional customization options:
#     # spacing_size=gr.themes.sizes.spacing_md,  # Medium spacing
#     # radius_size=gr.themes.sizes.radius_md,    # Medium border radius
#     # text_size=gr.themes.sizes.text_md,        # Medium text size
# )

# Create the main Gradio interface with custom theme and CSS
# This sets up the web interface with our custom styling and layout
with gr.Blocks(title="Local Ollama RAG Resume Assistant", theme=theme, css="""
    /* Custom CSS for enhanced styling and Nunito font consistency */
    .tab-header {
        margin-bottom: 30px !important;  /* More space below tab headers */
        font-family: 'Nunito', sans-serif !important;  /* Consistent Nunito font */
        font-weight: 500 !important;  /* Medium font weight for headers */
    }
    .gradio-container {
        max-width: 1200px !important;  /* Limit container width for better readability */
        margin: 0 auto !important;  /* Center the container */
    }
    .gr-button {
        font-family: 'Nunito', sans-serif !important;  /* Nunito font for buttons */
        font-weight: 500 !important;  /* Medium font weight */
        padding: 12px 24px !important;  /* More padding for better touch targets */
    }
    .gr-textbox, .gr-file {
        font-family: 'Nunito', sans-serif !important;  /* Nunito font for input fields */
    }
    .gr-markdown {
        font-family: 'Nunito', sans-serif !important;  /* Nunito font for markdown content */
        line-height: 1.6 !important;  /* Better line spacing for readability */
    }
    .critique-button {
        margin-bottom: 20px !important;  /* Add space below the button */
        font-size: 16px !important;      /* Larger button text */
        padding: 14px 28px !important;   /* More padding for better appearance */
    }
    .critique-output {
        min-height: 600px !important;  /* Make critique output much taller */
        max-height: 800px !important;  /* Set maximum height */
        overflow-y: auto !important;   /* Allow scrolling if content is very long */
        padding: 20px !important;      /* Add padding for better spacing */
        border: 1px solid #e2e8f0 !important;  /* Add subtle border */
        border-radius: 12px !important;  /* Rounded corners */
        background-color: #ffffff !important;  /* White background for better contrast */
        font-size: 16px !important;    /* Larger font size for better readability */
        color: #1f2937 !important;     /* Dark text color for visibility */
    }
    .critique-output h1, .critique-output h2, .critique-output h3 {
        color: #1f2937 !important;     /* Dark color for headings */
    }
    .critique-output p, .critique-output li {
        color: #374151 !important;     /* Slightly lighter dark color for body text */
    }
    .critique-output strong {
        color: #1f2937 !important;     /* Dark color for bold text */
    }
    .critique-output * {
        color: inherit !important;     /* Ensure all child elements inherit the dark color */
    }
    .critique-output span {
        color: inherit !important;     /* Ensure spans inherit the proper color */
    }
""") as demo:
    # Create the main header with app title and description
    # This provides a welcoming introduction to the application
    gr.HTML(f"""
        <div style="text-align: center; max-width: 900px; margin: 0 auto; padding: 40px 30px; background-color: #f8fafc; border-radius: 20px; border: 1px solid #e2e8f0; font-family: 'Nunito', sans-serif;">
            <h1 style="font-size: 2.5em; color: #1e40af; font-weight: 600; margin-bottom: 20px; letter-spacing: -0.02em;">Local RAG Resume Assistant</h1>
            <p style="color: #64748b; font-size: 1.2em; line-height: 1.6; margin: 0; font-weight: 400;">
                Powered by LangChain, Ollama ({LLM_MODEL} & {EMBEDDING_MODEL}), and ChromaDB.<br>
                Upload your resume (PDF/DOCX) below, and then ask questions or request a professional critique.
            </p>
        </div>
    """)

    # Create the main tabbed interface
    # Each tab represents a different step in the resume analysis workflow
    with gr.Tabs():
        
        # 1. UPLOAD AND INDEX TAB - First step: Upload and process resume
        with gr.TabItem("📄 Upload & Index Resume"):
            gr.Markdown("### Step 1: Upload Your Resume", elem_classes="tab-header")
            
            # File Upload Component - Accepts common resume formats
            file_upload = gr.File(
                label="Upload Resume (PDF, DOCX, TXT recommended)",
                file_types=[".pdf", ".docx", ".txt"]  # Supported file formats
            )
            
            # Button to trigger the resume processing pipeline
            process_button = gr.Button("Process and Index Resume", variant="primary")
            
            # Status display for processing feedback
            ingestion_status = gr.Textbox(
                label="Processing Status", 
                interactive=False,  # Read-only status display
                value="Please upload a file and click 'Process and Index Resume'."
            )
            
            # Connect the button click to the resume processing function
            process_button.click(
                fn=ingest_resume,  # Function to call
                inputs=[file_upload],  # Input from file upload
                outputs=[ingestion_status]  # Output to status display
            )
            
        # 2. Q&A CHAT TAB - Second step: Interactive Q&A about the resume
        with gr.TabItem("💬 Ask Q&A"):
            gr.Markdown("### Step 2: Ask Questions About Your Resume", elem_classes="tab-header")
            
            # Chat Interface - Provides conversational interaction with resume data
            # Uses RAG to answer questions based on the uploaded resume content
            chatbot = gr.ChatInterface(
                fn=ask_question,  # Function that handles the RAG pipeline
                textbox=gr.Textbox(
                    placeholder="E.g., What are my three main skills?",  # Helpful example
                    container=False  # Remove container for cleaner look
                ),
                title=f"Chat with Resume Data (LLM: {LLM_MODEL})"  # Show which model is being used
            )

        # 3. CRITIQUE TAB - Third step: Get professional resume feedback
        with gr.TabItem("✨ Get Critique"):
            gr.Markdown("### Step 3: Get a Structured Critique and Suggestions", elem_classes="tab-header")
            
            # Button to generate comprehensive resume critique
            critique_button = gr.Button("Generate Professional Critique", variant="secondary", elem_classes="critique-button")
            
            # Display area for the generated critique - made larger for better readability
            critique_output = gr.Markdown(
                "Critique will appear here after clicking the button. Ensure a resume has been successfully processed first.",
                elem_classes="critique-output"  # Add custom class for styling
            )
            
            # Connect button click to critique generation function
            critique_button.click(
                fn=critique_resume,  # Function that generates structured critique
                inputs=None,  # No inputs needed - uses global resume data
                outputs=[critique_output]  # Output to the markdown display
            )

# Application entry point
if __name__ == "__main__":
    # Launch the Gradio web application
    # This starts the local web server and opens the interface in the browser
    demo.launch()
