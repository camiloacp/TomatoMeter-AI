import streamlit as st
import pandas as pd
from utils import analyze_with_t5, chatgpt_generation

# ============================================
# BATCH PREDICTION FUNCTIONS
# ============================================
def predict_batch_pipeline(df, text_column, model):
    """Predicciones en batch con Pipeline Model"""
    predictions = []
    progress_bar = st.progress(0)
    
    for idx, text in enumerate(df[text_column]):
        result = model(text)[0]
        sentiment = max(result, key=lambda x: x['score'])
        predictions.append({
            'Sentiment': sentiment['label'].title(),
            'Confidence': sentiment['score']
        })
        progress_bar.progress((idx + 1) / len(df))
    
    return pd.DataFrame(predictions)

def predict_batch_t5(df, text_column, model):
    """Predicciones en batch con FLAN-T5"""
    predictions = []
    progress_bar = st.progress(0)
    
    for idx, text in enumerate(df[text_column]):
        result = analyze_with_t5(text, model)
        predictions.append({
            'Sentiment': result['label'].title(),
            'Confidence': result['score']
        })
        progress_bar.progress((idx + 1) / len(df))
    
    return pd.DataFrame(predictions)

def predict_batch_lr(df, text_column, lr_model, model):
    """Predicciones en batch con Logistic Regression"""
    # Asume que tu modelo LR tiene un método predict y predict_proba
    predictions = lr_model.predict(model.encode(df[text_column]))
    probabilities = lr_model.predict_proba(model.encode(df[text_column]))
    
    results = pd.DataFrame({
        'Sentiment': ['Positive' if p == 1 else 'Negative' for p in predictions],
        'Confidence': probabilities.max(axis=1)
    })
    
    return results

def predict_batch_chatgpt(df, text_column, model_name):
    """Predicciones en batch con ChatGPT (cuidado con rate limits)"""
    predictions = []
    progress_bar = st.progress(0)
    
    prompt = """Analyze the sentiment of this movie review. 
    Respond with only 'POSITIVE' or 'NEGATIVE'.
    
    Review: [DOCUMENT]"""
    
    for idx, text in enumerate(df[text_column]):
        try:
            result = chatgpt_generation(prompt, text, model_name)
            sentiment = 'POSITIVE'.title() if 'positive' in result.lower() else 'NEGATIVE'.title()
            predictions.append({
                'Sentiment': sentiment,
                'Confidence': 0.9  # ChatGPT no da scores directamente
            })
        except Exception:
            predictions.append({
                'Sentiment': 'ERROR',
                'Confidence': 0.0
            })
        
        progress_bar.progress((idx + 1) / len(df))
    
    return pd.DataFrame(predictions)