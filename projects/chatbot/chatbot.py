import os
from pathlib import Path
from cmath import nan
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from numpy import nan
import streamlit as st
import pandas as pd
# Constants
PATHS = {
    "ROLES_CSV": "./projects/chatbot/resources/roles.csv",
    "CHAT_SAVES_DIR": "./projects/chatbot/chat_saves/"
}
MODEL_CONFIG = {
    "DEFAULT_MODEL": "qwen3:4b-instruct",
    "AVAILABLE_MODELS": ["qwen3:4b-instruct", " qwen3.5-9b"],
    "DEFAULT_TEMPERATURE": 0.7,
    "TEMPERATURE_RANGE": (0.0, 1.0),
    "TEMPERATURE_STEP": 0.1,
    "CONTEXT_WINDOW_SIZE": 16384
}
UI_CONSTANTS = {
    "PAGE_TITLE": "Chatbot",
    "PAGE_ICON": "🤖",
    "LAYOUT": "wide",
    "HEADER_TEXT": "chatbot",
    "SIDEBAR_HEADER": "Model Parameters",
    "SIDEBAR_WIDTH": 150,
    "TEXT_AREA_HEIGHT": 400,
    "CHAT_INPUT_PLACEHOLDER": "Type your message here...",
    "SAVE_BUTTON_LABEL": "Save Chat History",
    "CLEAR_BUTTON_LABEL": "New Chat",
}
SESSION_STATE_KEYS = {
    "CHAT_SESSION_ID": "chat_session_id",
    "SYSTEM_PROMPT_KEY": "sys_prompt",
    "ROLE_SELECTION_KEY": "role_selection",
    "CHAT_HISTORY_KEY": "chat_history",
    "REASONING_KEY": "reasoning_enabled"
}
SYSTEM_VALUES = {
    "CHUNK_SIZE_FOR_PRE_SAVE_HISTORY": 200
}

## CALLBACKS for ui components
def clear_chat_history():
    st.session_state.pop(SESSION_STATE_KEYS["CHAT_HISTORY_KEY"], None)
    st.session_state.pop(SESSION_STATE_KEYS["CHAT_SESSION_ID"], None)
    
def onchange_sys_prompt():
    """Callback when system prompt changes."""
    st.session_state[SESSION_STATE_KEYS["ROLE_SELECTION_KEY"]] = None
    
def save_chat_history(partial_history=None):
    """Save chat history to file."""
    ensure_chat_saves_dir()
    
    sys_prompt = st.session_state.get(SESSION_STATE_KEYS["SYSTEM_PROMPT_KEY"], "")
    chat_history = partial_history if partial_history is not None else st.session_state.get(SESSION_STATE_KEYS["CHAT_HISTORY_KEY"], [])
    
    timestamp = st.session_state.get(SESSION_STATE_KEYS["CHAT_SESSION_ID"], str(pd.Timestamp.now().timestamp()))
    filename = f"chat_history_{timestamp}.md"
    filepath = os.path.join(PATHS["CHAT_SAVES_DIR"], filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"System Prompt:\n{sys_prompt}\n\n")
        for msg in chat_history:
            f.write(f"{msg.type}: {msg.content}\n")

def on_reset_role_selection():
    """Callback to reset role selection."""
    st.session_state.update({SESSION_STATE_KEYS["ROLE_SELECTION_KEY"]: None})
    st.session_state.update({SESSION_STATE_KEYS["SYSTEM_PROMPT_KEY"]: ""})
    
    
## Helper functions
def ensure_chat_saves_dir():
    """Ensure the chat saves directory exists."""
    Path(PATHS["CHAT_SAVES_DIR"]).mkdir(parents=True, exist_ok=True)
def load_roles():
    """Load roles from CSV file."""
    return pd.read_csv(PATHS["ROLES_CSV"], sep=",").to_dict(orient="records")
def format_role(role):
    """Format role for display in selectbox."""
    return f"{role['id']}: {role['name']}"

def init_session_state():
    """Initialize session state variables."""
    if SESSION_STATE_KEYS["CHAT_SESSION_ID"] not in st.session_state:
        st.session_state[SESSION_STATE_KEYS["CHAT_SESSION_ID"]] = str(pd.Timestamp.now().timestamp())
    if SESSION_STATE_KEYS["CHAT_HISTORY_KEY"] not in st.session_state:
        st.session_state[SESSION_STATE_KEYS["CHAT_HISTORY_KEY"]] = []
    if SESSION_STATE_KEYS["ROLE_SELECTION_KEY"] not in st.session_state:
        st.session_state[SESSION_STATE_KEYS["ROLE_SELECTION_KEY"]] = None
    if SESSION_STATE_KEYS["REASONING_KEY"] not in st.session_state:
        st.session_state[SESSION_STATE_KEYS["REASONING_KEY"]] = False
        
def generate_system_prompt(role_select):
    return role_select["prompt"].strip().replace("\\n", "\n")

    
def display_chat_history():
    """Display chat history messages."""
    chat_history = st.session_state.get(SESSION_STATE_KEYS["CHAT_HISTORY_KEY"], [])
    for msg in chat_history:
        with st.chat_message(msg.type):
            st.markdown(msg.content)
def handle_user_input(sys_txt, model_selected, temperature, is_reasoning):
    """Handle user input and generate AI response."""
    user_input = st.chat_input(UI_CONSTANTS["CHAT_INPUT_PLACEHOLDER"])
    
    if user_input:
        # Display user message
        with st.chat_message("human"):
            st.markdown(user_input)
        
        # Generate AI response
        full_response = ""
        text = ""
        reasoning = ""
        m = st.empty()
        e = st.empty()
        reasoning_message = st.empty()
        new_message = st.empty()
        
        # Create prompt template
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", sys_txt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])
        
        # Format prompt with history and input
        chat_history = st.session_state.get(SESSION_STATE_KEYS["CHAT_HISTORY_KEY"], [])
        full_chat_prompt = chat_prompt.format(history=chat_history, input=user_input)
        
        # Initialize model and stream response
        model = ChatOllama(model=model_selected, temperature=temperature, reasoning=is_reasoning, num_ctx=MODEL_CONFIG["CONTEXT_WINDOW_SIZE"])
        
        count = 0
        for token in model.stream(full_chat_prompt):
            count += 1
            block  = token.content_blocks[0] if token.content_blocks else None

            if block and block["type"] == "reasoning":
                reasoning += block["reasoning"]
            if block and block["type"] == "text":
                text += block["text"]
                
            if reasoning:
               full_response =  f"""<div style="color: darkgray; font-size: 14px;">[Tinking]: {reasoning}</div>"""
            if text and reasoning:
                full_response += f"\n\n{text}"
            elif text:
                full_response = text
            
            if count >= SYSTEM_VALUES["CHUNK_SIZE_FOR_PRE_SAVE_HISTORY"]:
                count = 0
                partial_history = []
                partial_history.extend(chat_history)
                partial_history.append(HumanMessage(user_input))
                partial_history.append(AIMessage(full_response))
                save_chat_history(partial_history)
            
            #full_response += token.content
            new_message.markdown(full_response + "⋯", unsafe_allow_html=True)
        
        with m.chat_message("ai"):
            if reasoning:
                e.expander("Reasoning").markdown(reasoning)
                #reasoning_message.markdown(f"[Think]: {reasoning}")
            new_message.markdown(text)
        
        # Update chat history
        chat_history.append(HumanMessage(user_input))
        chat_history.append(AIMessage(full_response))
        st.session_state[SESSION_STATE_KEYS["CHAT_HISTORY_KEY"]] = chat_history
        save_chat_history()

            
def create_sidebar(roles):
    """Create and render sidebar with controls."""
    with st.sidebar:
        st.header(UI_CONSTANTS["SIDEBAR_HEADER"])
        
        left, right = st.columns(2, gap="small",vertical_alignment="center",)
        with left:
            model_selected = st.selectbox(
                "Select Model:",
                options=[model for model in MODEL_CONFIG["AVAILABLE_MODELS"]],
                index=0,
                key="model_selection"
            )
        with right:
            reasoning = st.checkbox("Reasoning", key="reasoning_checkbox", value=st.session_state.get(SESSION_STATE_KEYS["REASONING_KEY"], False))
            
        # Temperature slider
        temp = st.slider(
            "Temperature",
            min_value=MODEL_CONFIG["TEMPERATURE_RANGE"][0],
            max_value=MODEL_CONFIG["TEMPERATURE_RANGE"][1],
            value=MODEL_CONFIG["DEFAULT_TEMPERATURE"],
            step=MODEL_CONFIG["TEMPERATURE_STEP"]
        )
        
        # Role selection
        role_select = st.selectbox(
            "Select Role:",
            options=roles,
            format_func=format_role,
            index=0,
            key=SESSION_STATE_KEYS["ROLE_SELECTION_KEY"]
        )
        st.button("Reset Role Selection", on_click=on_reset_role_selection, type="secondary", use_container_width=True)
        sys_txt = ""
        # System prompt handling
        if role_select:
            # Generate prompt based on role
            sys_prompt = generate_system_prompt(role_select)
            st.session_state[SESSION_STATE_KEYS["SYSTEM_PROMPT_KEY"]] = sys_prompt
            
            # Text area with generated prompt as default
            sys_txt = st.text_area(
                "Enter system prompt:",
                key=SESSION_STATE_KEYS["SYSTEM_PROMPT_KEY"],
                on_change=onchange_sys_prompt,
                value=st.session_state.get(SESSION_STATE_KEYS["SYSTEM_PROMPT_KEY"], sys_prompt),
                height=UI_CONSTANTS["TEXT_AREA_HEIGHT"]
            )
        else:
            # Text area for custom prompt
            sys_txt = st.text_area(
                "Enter system prompt:",
                key=SESSION_STATE_KEYS["SYSTEM_PROMPT_KEY"],
                height=UI_CONSTANTS["TEXT_AREA_HEIGHT"],
                on_change=onchange_sys_prompt
            )
        
        # Action buttons
        left, right = st.columns(2)
        with left:
            st.button(
                UI_CONSTANTS["SAVE_BUTTON_LABEL"],
                key="save_chat_history",
                on_click=save_chat_history,
                type="secondary"
            )
        with right:
            st.button(
                UI_CONSTANTS["CLEAR_BUTTON_LABEL"],
                key="clear_chat_history",
                on_click=clear_chat_history,
                type="primary"
            )
        
        return sys_txt, temp, model_selected, reasoning
def main():
    """Main application function."""
    # Page configuration
    st.set_page_config(
        page_title=UI_CONSTANTS["PAGE_TITLE"],
        page_icon=UI_CONSTANTS["PAGE_ICON"],
        layout=UI_CONSTANTS["LAYOUT"]
    )
    st.header(UI_CONSTANTS["HEADER_TEXT"], anchor="top")
    
    # Initialize session state
    init_session_state()
    
    # Load roles
    roles = load_roles()
    
    # Create sidebar and get values
    sys_txt, temperature, model_selected, reasoning = create_sidebar(roles)
    
    # Display chat history
    display_chat_history()
    
    # Handle user input
    handle_user_input(sys_txt, model_selected, temperature, reasoning)
if __name__ == "__main__":
    main()