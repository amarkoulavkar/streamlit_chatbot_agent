from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os

def build_doc_retriever(uploaded_file):
    """
    Takes a Streamlit UploadedFile, saves it, loads the document, and returns a retriever.
    Returns None if file type is not supported.
    """
    file_path = f"temp_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif uploaded_file.name.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif uploaded_file.name.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        os.remove(file_path)
        return None
    docs = loader.load()
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever()
    os.remove(file_path)
    return retriever

def get_doc_answer(llm, retriever, user_input):
    """
    Given an LLM, a retriever, and a user question, returns an answer based on the document.
    """
    related_docs = retriever.get_relevant_documents(user_input)
    context = "\n".join([doc.page_content for doc in related_docs])
    prompt = f"Answer the question based on the following document context:\n{context}\n\nQuestion: {user_input}"
    return llm.invoke(prompt)
