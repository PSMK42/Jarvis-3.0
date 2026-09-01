import os
import streamlit as st
from groq import Groq
from streamlit_mic_recorder import speech_to_text

# Configure page layout
st.set_page_config(
    page_title="Jarvis AI",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Jarvis AI Interface (Groq)")
st.write("Talk to Jarvis using your voice or type a message below.")

# Explicitly retrieve API key from Streamlit Secrets
api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY is missing! Please add it to Streamlit Secrets.")
    st.stop()

client = Groq(api_key=api_key)

SYSTEM_PROMPT = (
    "You are Jarvis, a highly capable, concise, and helpful AI assistant. "
    "Respond naturally and keep your answers clear and direct."
)

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display full conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# --- VOICE INPUT SECTION ---
st.write("---")
st.subheader("🎤 Voice Control")

# Speech recognition widget
voice_text = speech_to_text(
    start_prompt="🎤 Click to Speak",
    stop_prompt="⏹️ Stop Recording",
    language='en',
    use_container_width=True,
    key='speech'
)

# Text input widget
typed_text = st.chat_input("Type your command here...")

# Pick whichever input source received data
prompt = voice_text or typed_text

# --- PROCESS INPUT ---
if prompt:
    # 1. Add user message to history and render it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 2. Get AI response from Groq
    with st.chat_message("assistant"):
        with st.spinner("Jarvis is thinking..."):
            try:
                # Prepare message payload with system prompt
                api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                api_messages.extend(st.session_state.messages)

                # Send request to Groq API
                chat_completion = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=api_messages,
                    temperature=0.7,
                    max_tokens=1024,
                )

                reply = chat_completion.choices[0].message.content
                st.write(reply)

                # 3. Add assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": reply})

            except Exception as e:
                st.error(f"Error communicating with Jarvis: {e}")
