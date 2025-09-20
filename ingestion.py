from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

def ingest_pdfs():
    # Get absolute path to college_documents folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_folder = os.path.join(current_dir, "college_documents")
    
    # Verify directory exists
    if not os.path.exists(pdf_folder):
        raise FileNotFoundError(
            f"PDF directory not found at: {pdf_folder}\n"
            f"Please create a 'college_documents' folder in the same directory as this script\n"
            f"and place your PDF files inside it."
        )

    # Load all PDFs from directory
    loader = DirectoryLoader(
        pdf_folder,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    docs = loader.load()
    
    if not docs:
        raise ValueError(f"No PDF documents found in {pdf_folder}")

    # Text splitting with proper regex
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", r"(?<=\. )", " "]
    )
    chunks = splitter.split_documents(docs)
    
    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=os.path.join(current_dir, "chroma_db")
    )
    
    print(f"✅ Successfully processed {len(docs)} pages from PDFs in {pdf_folder}")

if __name__ == "__main__":
    ingest_pdfs()