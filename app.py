import streamlit as st
import whisper
from googletrans import Translator, LANGUAGES
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

# Load Whisper model
model = whisper.load_model("base")
print("Model loaded successfully")

# Load smaller DistilGPT-2 model and tokenizer from Hugging Face
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("distilgpt2")

# Function to transcribe audio
def transcribe_audio(audio_file):
    file_path = "uploaded_audio.wav"
    with open(file_path, "wb") as f:
        f.write(audio_file.getbuffer())
    
    audio = whisper.load_audio(file_path)
    result = model.transcribe(audio)
    os.remove(file_path)
    return result['text']

# Function to translate text
def translate_text(text, target_lang):
    translator = Translator()
    translated = translator.translate(text, dest=target_lang)
    return translated.text

# Function to generate text using DistilGPT-2
def generate_text_with_gpt2(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = gpt2_model.generate(inputs, max_length=500, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Streamlit UI layout
st.set_page_config(page_title="Talkify AI", page_icon="🎤", layout="wide")

st.markdown("""
    <style>
    .header {
        font-size: 48px;
        color: #4A90E2;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    .subheader {
        font-size: 24px;
        color: #4A90E2;
        text-align: center;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<div class="header">🎤 Talkify AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Your Intelligent Voice Assistant 🤖💬</div>', unsafe_allow_html=True)

# Audio Upload Section
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

# Language selection
language_options = {
    'Arabic': 'ar', 'Chinese (Simplified)': 'zh-cn', 'English': 'en', 'French': 'fr',
    'German': 'de', 'Hindi': 'hi', 'Italian': 'it', 'Japanese': 'ja', 'Russian': 'ru', 'Spanish': 'es'
}
selected_language = st.selectbox("Choose the translation language:", list(language_options.keys()), index=2)

if audio_file is not None:
    st.audio(audio_file, format="audio/wav")

    # Transcribe audio
    with st.spinner('🔊 Transcribing audio...'):
        transcribed_text = transcribe_audio(audio_file)
    st.write("### 🎙️ Transcribed Text:", transcribed_text)

    # Translate the transcribed text
    with st.spinner(f'🌐 Translating text to {selected_language}...'):
        translated_text = translate_text(transcribed_text, language_options[selected_language])
    st.write(f"### 🗣️ Translated Text ({selected_language}):", translated_text)

    # Generate DistilGPT-2 response
    with st.spinner('🤖 Generating AI response...'):
        ai_response = generate_text_with_gpt2(transcribed_text)
    st.write("### 💬 AI Generated Response:", ai_response)

    # Add a button for restarting
    if st.button('🔄 Start Over'):
        st.experimental_rerun()
else:
    st.info('👆 Upload an audio file to begin!')

# Footer
st.markdown('---')
st.markdown('<div style="text-align: center; color: #888;">Made with ❤️ by Zain Mehmood</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center; color: #0073e6;">Check out the project on GitHub: <a href="https://github.com/ZainMehmood123/Talkify-AI" target="_blank">Talkify AI GitHub</a></div>', unsafe_allow_html=True)
