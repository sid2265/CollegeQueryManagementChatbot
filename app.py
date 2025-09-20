import streamlit as st
from rag import get_qa_chain

# Custom CSS for college theme
st.set_page_config(
    page_title="College Query Bot",
    page_icon="🏫",
    layout="wide"
)

@st.cache_resource
def load_chain():
    return get_qa_chain()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to CollegeBot! Ask about admissions, placements, or academics."}
    ]

# Display chat
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle queries
if prompt := st.chat_input("Ask a college question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    with st.spinner("Checking college documents..."):
        response = load_chain().invoke({"query": prompt})
        answer = response["result"]
        
        # Format sources from PDFs
        sources = []
        for doc in response["source_documents"]:
            if "source" in doc.metadata:
                src = doc.metadata["source"].split("/")[-1]
                sources.append(f"📄 {src} (page {doc.metadata.get('page', '?')})")
        
    # Display response
    with st.chat_message("assistant"):
        st.markdown(answer)
        if sources:
            st.caption("\n".join(set(sources)))  # Remove duplicates
    
    st.session_state.messages.append({"role": "assistant", "content": answer})