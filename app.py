import streamlit as st
import whisper
from googletrans import Translator, LANGUAGES
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import tempfile
import os

# Load Whisper model
model = whisper.load_model("base")
print("Model loaded successfully")


# Load GPT-2 model and tokenizer from Hugging Face
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

# Function to transcribe audio
def transcribe_audio(audio_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        tmpfile.write(audio_file.getbuffer())
        tmpfile_path = tmpfile.name
    audio = whisper.load_audio(tmpfile_path)
    result = model.transcribe(audio)
    os.remove(tmpfile_path)
    return result['text']

# Function to translate text
def translate_text(text, target_lang):
    translator = Translator()
    translated = translator.translate(text, dest=target_lang)
    return translated.text

# Function to generate text using GPT-2 from Hugging Face
def generate_text_with_gpt2(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = gpt2_model.generate(inputs, max_length=2500, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Streamlit UI layout with enhanced styling
st.set_page_config(page_title="Talkify AI", page_icon="🎤", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    body {
        background-color: #f0f5ff;
        font-family: 'Roboto', sans-serif;
        color: #333;
    }

    .header {
        font-size: 48px;
        color: #4A90E2;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    .subheader {
        font-size: 24px;
        color: #4A90E2;
        text-align: center;
        margin-bottom: 30px;
    }

    .section-header {
        font-size: 28px;
        color: #4A90E2;
        font-weight: bold;
        margin-top: 40px;
        margin-bottom: 20px;
    }

    .box {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }

    .box:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }

    .box-header {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 10px;
    }

    .transcribed-text {
        color: #1E88E5;
    }

    .translated-text {
        color: #039BE5;
    }

    .ai-response-text {
        color: #0288D1;
    }

    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 12px 24px;
        text-align: center;
        font-size: 18px;
        border-radius: 5px;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }

    .audio-upload-section {
        margin-top: 20px;
        font-size: 18px;
        text-align: center;
    }

    .emoji {
        font-size: 36px;
        margin-right: 10px;
        vertical-align: middle;
    }

    .stAudio {
        width: 100%;
    }

    .language-select {
        margin-top: 20px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<div class="header">🎤 Talkify AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Your Intelligent Voice Assistant 🤖💬</div>', unsafe_allow_html=True)

# Audio Upload Section
st.markdown('<div class="section-header">📤 Upload Your Audio</div>', unsafe_allow_html=True)
st.markdown('<div class="audio-upload-section">🎵 Select an audio file to get started!</div>', unsafe_allow_html=True)
audio_file = st.file_uploader("", type=["wav", "mp3"])

# Language selection
language_options = {
    'Arabic': 'ar',
    'Chinese (Simplified)': 'zh-cn',
    'English': 'en',
    'French': 'fr',
    'German': 'de',
    'Hindi': 'hi',
    'Italian': 'it',
    'Japanese': 'ja',
    'Russian': 'ru',
    'Spanish': 'es'
}

st.markdown('<div class="section-header">🌐 Select Translation Language</div>', unsafe_allow_html=True)
selected_language = st.selectbox(
    "Choose the language for translation:",
    options=list(language_options.keys()),
    index=2,  # Default to English
    key="language_select"
)

if audio_file is not None:
    st.audio(audio_file, format="audio/wav")

    # Transcribe audio
    with st.spinner('🔊 Transcribing audio...'):
        transcribed_text = transcribe_audio(audio_file)
    st.markdown('<div class="section-header">📝 Transcription Results</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="box"><div class="box-header transcribed-text">🎙️ Transcribed Text:</div><div class="transcribed-text">{transcribed_text}</div></div>', unsafe_allow_html=True)

    # Translate the transcribed text
    with st.spinner(f'🌐 Translating text to {selected_language}...'):
        translated_text = translate_text(transcribed_text, language_options[selected_language])
    st.markdown('<div class="section-header">🌍 Translation</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="box"><div class="box-header translated-text">🗣️ Translated Text ({selected_language}):</div><div class="translated-text">{translated_text}</div></div>', unsafe_allow_html=True)

    # Generate GPT-2 response
    with st.spinner('🤖 Generating AI response...'):
        ai_response = generate_text_with_gpt2(transcribed_text)
    st.markdown('<div class="section-header">🤖 AI Chatbot Response</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="box"><div class="box-header ai-response-text">💬 AI Generated Response:</div><div class="ai-response-text">{ai_response}</div></div>', unsafe_allow_html=True)

    # Add a button for restarting or clearing
    if st.button('🔄 Start Over'):
        st.experimental_rerun()

else:
    st.info('👆 Upload an audio file to begin!')

# Footer with GitHub link
st.markdown('---')
st.markdown('<div style="text-align: center; color: #888;">Made with ❤️ by Zain Mehmood</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center; color: #0073e6;">Check out the project on GitHub: <a href="https://github.com/ZainMehmood123/Talkify-AI" target="_blank">Talkify AI GitHub</a></div>', unsafe_allow_html=True)
