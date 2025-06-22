import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent

# Set your Gemini API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCvvm5VCVn2ur0TLD3_Sh_4mfzJf86bQ_E"

# Load LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Load tools
tools = load_tools(["llm-math"], llm=llm)

# Initialize Agent
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=False
)

st.title("🤖 LangChain AI Chatbot")

user_input = st.text_input("Ask me anything:")

if user_input:
    response = agent.invoke(user_input)
    st.write("🤖", response)