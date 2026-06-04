import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Load environment variables
load_dotenv()

# Streamlit App Layout
st.set_page_config(page_title="Gemini Chatbot", page_icon="🤖")
st.title("🤖 LangChain Gemini Chatbot")

# Initialize the Gemini model exactly as you provided
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.9)

# Initialize session state for your 'messages' array so it isn't lost on rerun
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a helpful assistant. Answer the question as best as you can.")
    ]

# Display past chat interactions (ignoring the initial SystemMessage)
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# Handle the user chat input box
if prompt := st.chat_input("Type your message here..."):
    
    # 1. Visual: Immediately show user text in the UI
    with st.chat_message("user"):
        st.write(prompt)
        
    # 2. Logic: Append HumanMessage directly to your stateful messages array
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    # 3. Logic: Invoke the model passing the historical messages array
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = model.invoke(st.session_state.messages)
            st.write(response.content)
            
    # 4. Logic: Append AIMessage back into the message array history
    st.session_state.messages.append(AIMessage(content=response.content))
