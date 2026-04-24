import streamlit as st
from langchain_ollama import ChatOllama
from langchain_endee import EndeeVectorStore
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# --- UI Setup ---
st.set_page_config(page_title="Mission Command: Local", page_icon="🎖️")
st.title("🎖️ Mission Command: Defense Exam Assistant")
st.caption("Status: Operational | Engine: Local Llama 3.2 | Database: Endee DB")

# --- Initialize Local AI (No Keys Needed) ---
@st.cache_resource
def load_local_brain():
    # Connects to the Ollama model you just pulled
    return ChatOllama(model="llama3.2", temperature=0)

@st.cache_resource
def load_embedder():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = load_local_brain()
embeddings = load_embedder()

# --- Endee DB Logic ---
def get_vector_store():
    try:
        return EndeeVectorStore(
            embedding=embeddings,
            index_name="mission_command_local",
            dimension=384
        )
    except Exception as e:
        return None

vector_store = get_vector_store()

# --- Manual Initialization ---
if st.button("🚀 Deploy Defense Knowledge Base"):
    knowledge_base = [
        "IMA Eligibility: Unmarried male graduates, age 19-24 years.",
        "Training Academy: Indian Military Academy (IMA) is located in Dehradun, Uttarakhand.",
        "CDS Exam Structure: Consists of English, General Knowledge, and Elementary Mathematics.",
        "SSB Interview: A 5-day personality testing process following the written exam."
    ]
    docs = [Document(page_content=t) for t in knowledge_base]
    
    if vector_store:
        vector_store.add_documents(docs)
        st.success("Endee DB Synced with Tactical Data!")
    else:
        st.info("System Ready (Running in Local Fallback mode).")

# --- Chat Interface ---
query = st.chat_input("Ask about CDS syllabus, eligibility, or IMA training...")

if query:
    st.chat_message("user").write(query)
    
    with st.spinner("Analyzing local intelligence..."):
        # 1. Retrieval
        context = "The IMA is in Dehradun. Eligibility is 19-24 years." # Fallback
        if vector_store:
            results = vector_store.similarity_search(query, k=2)
            context = "\n".join([r.page_content for r in results])

        # 2. Local Generation
        prompt = f"""
        You are a helpful military exam advisor. Use the context below to answer.
        Context: {context}
        Question: {query}
        """
        response = llm.invoke(prompt)
        st.chat_message("assistant").write(response.content)