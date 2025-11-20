import os
import pickle

import streamlit as st
from datasets import load_dataset
from transformers import pipeline
from dotenv import load_dotenv
import openai
import numpy as np

from settings import MODEL_PATH_PIPELINE, MODEL_PATH_ENCODER_DECODER, MODEL_NAME_CHATGPT

load_dotenv()

st.set_page_config(
    page_title="TomatoMeter AI",
    page_icon="🍅",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================
# CUSTOM CSS STYLES
# ============================================
st.markdown("""
    <style>
    /* Importar fuente moderna */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    /* Aplicar fuente global */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header principal */
    .main-header {
        font-size: 3.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF6347 0%, #FF4500 50%, #DC143C 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
        margin-bottom: 0;
        letter-spacing: -1px;
    }
    
    .sub-header {
        text-align: center;
        color: #666666;
        margin-bottom: 3rem;
        font-size: 1.15rem;
        font-weight: 400;
    }
    
    /* Mejorar selectbox */
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 2px solid #f0f0f0;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #FF6347;
        box-shadow: 0 2px 8px rgba(255, 99, 71, 0.15);
    }
    
    /* Mejorar text area */
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #f0f0f0;
        padding: 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #FF6347;
        box-shadow: 0 4px 12px rgba(255, 99, 71, 0.2);
    }
    
    /* Mejorar info boxes */
    .stAlert {
        border-radius: 10px;
        border: none;
        padding: 0.75rem 1rem;
        font-weight: 500;
        margin-top: -8px !important;
    }
    
    /* Títulos de sección mejorados */
    h3 {
        color: #1a1a1a;
        font-weight: 700;
        margin-top: 2.5rem;
        margin-bottom: 1.5rem;
        font-size: 1.4rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #f0f0f0;
    }
    
    /* Emoji en títulos */
    h3::before {
        font-size: 1.5rem;
    }
    
    /* Espaciado general */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 800px;
    }
    
    /* Forzar alineación de columnas */
    div[data-testid="column"] > div {
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
    }
    
    div[data-testid="column"]:nth-child(2) .stAlert {
        margin-top: -10px !important;
    }
    
    /* Eliminar espacios extra */
    .stSelectbox {
        margin-bottom: 0 !important;
    }
    
    div[data-testid="column"]:nth-child(2) > div > div {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# LOAD FUNCTIONS
# ============================================
@st.cache_resource
def load_data():
    return load_dataset("rotten_tomatoes")

@st.cache_resource
def load_pipeline():
    return pipeline(
        model=MODEL_PATH_PIPELINE,
        tokenizer=MODEL_PATH_PIPELINE,
        return_all_scores=True,
        device="mps"
    )

@st.cache_resource
def load_lr_model():
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "model_lr.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_encoder_decoder_model():
    return pipeline(
        model=MODEL_PATH_ENCODER_DECODER,
        tokenizer=MODEL_PATH_ENCODER_DECODER,
        return_all_scores=True,
        device="mps"
    )

def chatgpt_generation(prompt, document, model=MODEL_NAME_CHATGPT):
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": prompt.replace("[DOCUMENT]", document)
        }
    ]
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model
    )
    return chat_completion.choices[0].message.content

# ============================================
# HEADER
# ============================================
st.markdown('<h1 class="main-header">🍅 TomatoMeter AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Discover the sentiment of your movie reviews with AI</p>', unsafe_allow_html=True)

# ============================================
# MODEL SELECTION
# ============================================
st.markdown("<h3>🤖 Select your AI Model</h3>", unsafe_allow_html=True)

col1, col2 = st.columns([3, 2], gap="medium")

with col1:
    model_option = st.selectbox(
        "Choose model",
        options=[
            "Hugging Face Pipeline",
            "Logistic Regression",
            "Encoder-Decoder - T5",
            "ChatGPT - GPT-4",
        ],
        index=0,
        label_visibility="collapsed"
    )

with col2:
    if model_option == "Hugging Face Pipeline":
        st.info("🚀 Fast & Easy")
    elif model_option == "Logistic Regression":
        st.success("⚡ Fastest")
    elif model_option == "Encoder-Decoder - T5":
        st.warning("🔄 Advanced")
    elif model_option == "ChatGPT - GPT-4":
        st.error("⭐ Premium")

# ============================================
# REVIEW INPUT
# ============================================
st.markdown("### 📝 Your Review")

review_text = st.text_area(
    "Review",
    height=160,
    placeholder="Example: This movie was absolutely amazing! The acting was superb and the plot kept me engaged throughout...",
    label_visibility="collapsed"
)

if st.button("Analyze Sentiment", type="primary", use_container_width=True):
    if review_text.strip():
        if model_option == "Hugging Face Pipeline":
            st.info("Analyzing sentiment with Hugging Face Pipeline...")
            pipe = load_pipeline()
            result = pipe(review_text)
            negative_score = result[0][0]["score"]
            positive_score = result[0][2]["score"]
            assignment = np.argmax([negative_score, positive_score])
            if assignment == 0:
                sentiment = "Negative"
                color = "red"
            else:
                sentiment = "Positive"
                color = "green"
            
            st.markdown(
                f"""
                <div style="padding: 1rem; border-radius: 0.5rem; background-color: {'#ffebee' if color == 'red' else '#e8f5e9'}; border-left: 5px solid {color};">
                    <h3 style="color: {color}; margin: 0;">Sentiment: {sentiment}</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

        elif model_option == "Logistic Regression":
            st.info("Analyzing sentiment with Logistic Regression...")
            lr_model = load_lr_model()
            
            #st.success(f"Sentiment: {result}")
        elif model_option == "Encoder-Decoder - T5":
            st.info("Analyzing sentiment with Encoder-Decoder - T5...")
            #result = encoder_decoder_sentiment(review_text)