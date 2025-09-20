from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

# College-specific prompt template
COLLEGE_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are an official college advisor. Answer strictly based on these documents:
    {context}

    Question: {question}
    Answer in clear, formal language. If unsure, say "Please contact the college administration".
    """,
)

def get_qa_chain():
    # Load vector DB with optimized settings
    current_dir = os.path.dirname(os.path.abspath(__file__))
    vector_db = Chroma(
        persist_directory=os.path.join(current_dir, "chroma_db"),
        embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        collection_metadata={"hnsw:space": "cosine"}
    )

    # Configure Ollama LLM
    llm = OllamaLLM(
        model="phi3",
        temperature=0.2,
        top_k=30,
        num_ctx=4096,
        system="You are a helpful college administrator assistant."
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(
            search_type="mmr",  # Max marginal relevance for better diversity
            search_kwargs={"k": 5, "score_threshold": 0.7}
        ),
        chain_type_kwargs={"prompt": COLLEGE_PROMPT},
        return_source_documents=True
    )