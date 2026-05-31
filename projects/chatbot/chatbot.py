
from cmath import nan

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from numpy import nan
from streamlit import streamlit as st
import pandas as pd

def format_role(role):
    return f"{role['id']}: {role['name']}"

def onchange_sys_prompt():
    st.session_state.role_selection = None

def save_chat_history():
    sys_prompt = st.session_state.get("sys_prompt", "")
    chat_history = st.session_state.get("chat_history", [])
    path = "./projects/chatbot/chat_saves/"
    filename = f"chat_history_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(path + filename, "w", encoding="utf-8") as f:
        f.write(f"System Prompt:\n{sys_prompt}\n\n")
        for msg in chat_history:
            f.write(f"{msg.type}: {msg.content}\n")
    
st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="wide")
st.header("chatbot", anchor="top", )
roles = pd.read_csv("./projects/chatbot/resources/roles.csv",sep=",").to_dict(orient="records")

with st.sidebar:
    st.header("Model Parameters", width=150)
    role_select = st.selectbox("Select Role:", options=roles, format_func=format_role, index=0, key="role_selection")
    
    if role_select and role_select["id"] != 0:
        output_format = f"\n\n# Output format:\n{role_select['output_format']}" if not pd.isna(role_select["output_format"]) and (role_select["output_format"] != "" or role_select["output_format"] is not None) else ""
        st.session_state["sys_prompt"] = f"""# Rol\n {role_select['rol']} \n\n# Propósito\n {role_select['proposito']} \n\n# Habilidades\n {role_select['habilidades']} {output_format}"""
        sys_txt = st.text_area("Enter system prompt:", key="sys_prompt", on_change=onchange_sys_prompt, value=st.session_state.get("sys_prompt", "eres un poderoso asistente de IA, responde simpre de manera breve resumida y concreta, sin explicaciones adicionales, a menos que se te pida lo contrario."), height=400)
    else:
        sys_txt = st.text_area("Enter system prompt:", key="sys_prompt", height=400, on_change=onchange_sys_prompt)
    print ("systxt: " + sys_txt)

    temp = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    left, right = st.columns(2)
    with left:
        st.button("Save Chat History", key="save_chat_history", on_click=save_chat_history, type="secondary")
    with right:
        st.button("Clear Chat History",key="clear_chat_history", on_click=lambda: st.session_state.pop("chat_history", None), type="primary")
    
        
chat_history = st.session_state.get("chat_history", [])

if chat_history:
    for i, (msg) in enumerate(chat_history):
        with st.chat_message(msg.type):
            st.markdown(msg.content)

user_input = st.chat_input("Type your message here...")

if user_input:
    with st.chat_message("human"):
        st.markdown(user_input)

    full_response = ""
    new_message = st.empty()
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", sys_txt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )
    full_chat_prompt = chat_prompt.format(history=chat_history, input=user_input)
    #print (full_chat_prompt)
    
    model = ChatOllama(model="qwen3:4b-instruct", temperature=temp)
    
    for s in model.stream(full_chat_prompt):
        full_response += s.content
        new_message.markdown(full_response + "▮")
    new_message.markdown(full_response)
        

    chat_history.append(HumanMessage(user_input))
    chat_history.append(AIMessage(full_response))
    st.session_state["chat_history"] = chat_history        