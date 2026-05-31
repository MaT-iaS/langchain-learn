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
    "AVAILABLE_MODELS": ["qwen3:4b-instruct", "qwen2.5-coder:7b"],
    "DEFAULT_TEMPERATURE": 0.7,
    "TEMPERATURE_RANGE": (0.0, 1.0),
    "TEMPERATURE_STEP": 0.1
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
    "CLEAR_BUTTON_LABEL": "Clear Chat History",
    "SYSTEM_PROMPT_KEY": "sys_prompt",
    "ROLE_SELECTION_KEY": "role_selection",
    "CHAT_HISTORY_KEY": "chat_history"
}
DEFAULT_VALUES = {
    "SYSTEM_PROMPT": "eres un poderoso asistente de IA, responde siempre de manera breve resumida y concreta, sin explicaciones adicionales, a menos que se te pida lo contrario."
}
def ensure_chat_saves_dir():
    """Ensure the chat saves directory exists."""
    Path(PATHS["CHAT_SAVES_DIR"]).mkdir(parents=True, exist_ok=True)
def load_roles():
    """Load roles from CSV file."""
    return pd.read_csv(PATHS["ROLES_CSV"], sep=",").to_dict(orient="records")
def format_role(role):
    """Format role for display in selectbox."""
    return f"{role['id']}: {role['name']}"
def onchange_sys_prompt():
    """Callback when system prompt changes."""
    st.session_state[UI_CONSTANTS["ROLE_SELECTION_KEY"]] = None
def init_session_state():
    """Initialize session state variables."""
    if UI_CONSTANTS["CHAT_HISTORY_KEY"] not in st.session_state:
        st.session_state[UI_CONSTANTS["CHAT_HISTORY_KEY"]] = []
    if UI_CONSTANTS["ROLE_SELECTION_KEY"] not in st.session_state:
        st.session_state[UI_CONSTANTS["ROLE_SELECTION_KEY"]] = None
    if UI_CONSTANTS["SYSTEM_PROMPT_KEY"] not in st.session_state:
        st.session_state[UI_CONSTANTS["SYSTEM_PROMPT_KEY"]] = DEFAULT_VALUES["SYSTEM_PROMPT"]
def generate_system_prompt(role_select):
    """Generate system prompt based on selected role."""
    if not role_select or role_select["id"] == 0:
        return DEFAULT_VALUES["SYSTEM_PROMPT"]
    
    output_format = ""
    if not pd.isna(role_select["output_format"]) and role_select["output_format"] != "":
        output_format = f"\n\n# Output format:\n{role_select['output_format']}"
    
    return f"""# Rol\n {role_select['rol']} \n\n# Propósito\n {role_select['proposito']} \n\n# Habilidades\n {role_select['habilidades']} {output_format}"""
def display_chat_history():
    """Display chat history messages."""
    chat_history = st.session_state.get(UI_CONSTANTS["CHAT_HISTORY_KEY"], [])
    for msg in chat_history:
        with st.chat_message(msg.type):
            st.markdown(msg.content)
def handle_user_input(sys_txt, model_selected, temperature):
    """Handle user input and generate AI response."""
    user_input = st.chat_input(UI_CONSTANTS["CHAT_INPUT_PLACEHOLDER"])
    
    if user_input:
        # Display user message
        with st.chat_message("human"):
            st.markdown(user_input)
        
        # Generate AI response
        full_response = ""
        new_message = st.empty()
        
        # Create prompt template
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", sys_txt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])
        
        # Format prompt with history and input
        chat_history = st.session_state.get(UI_CONSTANTS["CHAT_HISTORY_KEY"], [])
        full_chat_prompt = chat_prompt.format(history=chat_history, input=user_input)
        
        # Initialize model and stream response
        model = ChatOllama(model=model_selected, temperature=temperature)
        
        for s in model.stream(full_chat_prompt):
            full_response += s.content
            new_message.markdown(full_response + "▮")
        new_message.markdown(full_response)
        
        # Update chat history
        chat_history.append(HumanMessage(user_input))
        chat_history.append(AIMessage(full_response))
        st.session_state[UI_CONSTANTS["CHAT_HISTORY_KEY"]] = chat_history
def save_chat_history():
    """Save chat history to file."""
    ensure_chat_saves_dir()
    
    sys_prompt = st.session_state.get(UI_CONSTANTS["SYSTEM_PROMPT_KEY"], "")
    chat_history = st.session_state.get(UI_CONSTANTS["CHAT_HISTORY_KEY"], [])
    
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    filename = f"chat_history_{timestamp}.txt"
    filepath = os.path.join(PATHS["CHAT_SAVES_DIR"], filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"System Prompt:\n{sys_prompt}\n\n")
        for msg in chat_history:
            f.write(f"{msg.type}: {msg.content}\n")
def create_sidebar(roles):
    """Create and render sidebar with controls."""
    with st.sidebar:
        st.header(UI_CONSTANTS["SIDEBAR_HEADER"])
        
        model_selected = st.selectbox(
            "Select Model:",
            options=[model for model in MODEL_CONFIG["AVAILABLE_MODELS"]],
            index=0,
            key="model_selection"
        )
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
            key=UI_CONSTANTS["ROLE_SELECTION_KEY"]
        )
        
        # System prompt handling
        if role_select and role_select["id"] != 0:
            # Generate prompt based on role
            sys_prompt = generate_system_prompt(role_select)
            st.session_state[UI_CONSTANTS["SYSTEM_PROMPT_KEY"]] = sys_prompt
            
            # Text area with generated prompt as default
            sys_txt = st.text_area(
                "Enter system prompt:",
                key=UI_CONSTANTS["SYSTEM_PROMPT_KEY"],
                on_change=onchange_sys_prompt,
                value=st.session_state.get(UI_CONSTANTS["SYSTEM_PROMPT_KEY"], sys_prompt),
                height=UI_CONSTANTS["TEXT_AREA_HEIGHT"]
            )
        else:
            # Text area for custom prompt
            sys_txt = st.text_area(
                "Enter system prompt:",
                key=UI_CONSTANTS["SYSTEM_PROMPT_KEY"],
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
                on_click=lambda: st.session_state.pop(UI_CONSTANTS["CHAT_HISTORY_KEY"], None),
                type="primary"
            )
        
        return sys_txt, temp, model_selected
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
    sys_txt, temperature, model_selected = create_sidebar(roles)
    
    # Display chat history
    display_chat_history()
    
    # Handle user input
    handle_user_input(sys_txt, model_selected, temperature)
if __name__ == "__main__":
    main()