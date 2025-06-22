import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent
from document_qa import build_doc_retriever, get_doc_answer

# Set your Gemini API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCvvm5VCVn2ur0TLD3_Sh_4mfzJf86bQ_E"
# Set your SerpAPI key
os.environ["SERPAPI_API_KEY"] = "463f1d23cd502ea7961877989a8cdf2837438f031496831630b7f6f9875fb935"  # <-- Replace with your actual SerpAPI key

# Load LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Load tools (now includes web search)
tools = load_tools(["llm-math", "serpapi"], llm=llm)

# Initialize Agent
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=False
)

st.title("🤖 LangChain AI Chatbot")

st.markdown("### 📄 Document Q&A")
st.info("Upload a PDF, DOCX, or TXT file below. After uploading, ask questions about its content in the chat box.")

uploaded_file = st.file_uploader("Upload a document for Q&A:", type=["pdf", "docx", "txt"])
doc_retriever = None

if uploaded_file:
    doc_retriever = build_doc_retriever(uploaded_file)

st.markdown("---")
st.markdown("### 💬 Chatbot")

user_input = st.text_input("Ask me anything:")

if user_input:
    if doc_retriever:
        response = get_doc_answer(llm, doc_retriever, user_input)
        st.write("📄", response)
    else:
        response = agent.invoke(user_input)
        st.write("🤖", response)