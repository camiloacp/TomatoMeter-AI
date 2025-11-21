import streamlit as st
import pickle
from transformers import pipeline
from settings import MODEL_PATH_PIPELINE, MODEL_PATH_ENCODER_DECODER, MODEL_NAME_CHATGPT
import os
import openai

# ============================================
# CACHE FUNCTIONS
# ============================================
@st.cache_resource
def laod_pipeline():
    return pipeline(
        model=MODEL_PATH_PIPELINE,
        tokenizer=MODEL_PATH_PIPELINE,
        return_all_scores=True,
        device="mps"
    )

@st.cache_resource
def load_lr_model():
    with open("./model/model_lr.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_encoder_decoder_model():
    """Carga FLAN-T5 para generación de texto"""
    return pipeline(
        task="text2text-generation",
        model=MODEL_PATH_ENCODER_DECODER,
        tokenizer=MODEL_PATH_ENCODER_DECODER,
        device="mps"
    )

def analyze_with_t5(review_text, model):
    """Analiza sentimiento usando FLAN-T5"""
    prompt = f"""
    Classify the sentiment of this movie review as either 'positive' or 'negative'.
    Review: {review_text}
    
    Sentiment:"""
    
    result = model(prompt, max_length=10, num_return_sequences=1, do_sample=False)
    sentiment = result[0]['generated_text'].strip().lower()
    
    if 'positive' in sentiment:
        return {'label': 'POSITIVE', 'score': 0.95}
    elif 'negative' in sentiment:
        return {'label': 'NEGATIVE', 'score': 0.95}
    else:
        return {'label': 'NEUTRAL', 'score': 0.5}

def chatgpt_generation(prompt, document, model=MODEL_NAME_CHATGPT):
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant specialized in sentiment analysis."
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