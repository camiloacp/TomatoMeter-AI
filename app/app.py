import os
import pickle
import pandas as pd
from io import StringIO

import streamlit as st
from datasets import load_dataset
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import openai

from settings import MODEL_PATH_PIPELINE, MODEL_PATH_ENCODER_DECODER, MODEL_NAME_CHATGPT, MODEL_EMBEDDING

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
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #FF6347 0%, #FF4500 50%, #DC143C 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 0;
    }
    
    .sub-header {
        text-align: center;
        color: #888888;
        margin-bottom: 2rem;
        font-size: 1.2rem;
    }
    
    .upload-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #dee2e6;
        margin: 2rem 0;
    }
    
    .stDownloadButton {
        width: 100%;
    }
    
    .model-selector-header {
        font-size: 1.8rem;
        font-weight: 600;
        background: linear-gradient(135deg, #FF6347 0%, #FF4500 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding: 0.5rem 0;
        border-bottom: 2px solid #f0f0f0;
    }
""", unsafe_allow_html=True)

# ============================================
# CACHE FUNCTIONS
# ============================================
@st.cache_resource
def load_data():
    return load_dataset("rotten_tomatoes")

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
        except Exception as e:
            predictions.append({
                'Sentiment': 'ERROR',
                'Confidence': 0.0
            })
        
        progress_bar.progress((idx + 1) / len(df))
    
    return pd.DataFrame(predictions)

# ============================================
# UI - HEADER
# ============================================
st.markdown('<h1 class="main-header">🍅 TomatoMeter AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Discover the sentiment of your movie reviews</p>', unsafe_allow_html=True)

# ============================================
# MODEL SELECTOR
# ============================================
st.markdown('<h3 class="model-selector-header">🤖 Select your AI Model</h3>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])

with col1:
    model_option = st.selectbox(
        "Choose the model for sentiment analysis:",
        options=[
            "Pipeline Model (Transformer)",
            "Logistic Regression",
            "FLAN-T5 (Encoder-Decoder)",
            "ChatGPT"
        ],
        index=0,
        help="Select which AI model you want to use"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if model_option == "Pipeline Model (Transformer)":
        st.info("🚀 Fast")
    elif model_option == "Logistic Regression":
        st.success("⚡ Fastest")
    elif model_option == "FLAN-T5 (Encoder-Decoder)":
        st.warning("🔄 Advanced")
    else:
        st.error("🌟 Premium")

# ============================================
# TABS: SINGLE vs BATCH
# ============================================
tab1, tab2 = st.tabs(["📝 Single Review", "📊 Batch Analysis"])

# ============================================
# TAB 1: SINGLE REVIEW
# ============================================
with tab1:
    st.markdown("### ✍️ Write your review")
    
    review_text = st.text_area(
        "Enter your movie review here:",
        height=150,
        placeholder="Example: This movie was absolutely amazing! The acting was superb...",
        key="single_review"
    )
    
    if st.button("🔍 Analyze Sentiment", type="primary", key="analyze_single"):
        if review_text.strip():
            with st.spinner(f"Analyzing with {model_option}..."):
                try:
                    if model_option == "Pipeline Model (Transformer)":
                        model = laod_pipeline()
                        result = model(review_text)[0]
                        sentiment = max(result, key=lambda x: x['score'])
                        
                        st.success("✅ Analysis complete!")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Sentiment", sentiment['label'].title())
                        with col2:
                            st.metric("Confidence", f"{sentiment['score']:.2%}")
                        
                        # Mostrar todos los scores
                        with st.expander("📊 Detailed Scores"):
                            for item in result:
                                st.write(f"**{item['label']}**: {item['score']:.4f}")
                    
                    elif model_option == "Logistic Regression":
                        lr_model = load_lr_model()
                        
                        model = SentenceTransformer(MODEL_EMBEDDING)
                        embeddings = model.encode(review_text)
                        prediction = lr_model.predict([embeddings])[0]
                        proba = lr_model.predict_proba([embeddings])[0]
                        
                        st.success("✅ Analysis complete!")
                        sentiment_label = "Positive" if prediction == 1 else "Negative"
                        confidence = max(proba)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Sentiment", sentiment_label)
                        with col2:
                            st.metric("Confidence", f"{confidence:.2%}")
                    
                    elif model_option == "FLAN-T5 (Encoder-Decoder)":
                        model = load_encoder_decoder_model()
                        result = analyze_with_t5(review_text, model)
                        
                        st.success("✅ Analysis complete!")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Sentiment", result['label'].title())
                        with col2:
                            st.metric("Confidence", f"{result['score']:.2%}")
                    
                    else:  # ChatGPT
                        prompt = """
                        Analyze the sentiment of this movie review. 
                        Respond with 'POSITIVE' or 'NEGATIVE' and explain briefly why.
                        
                        Review: [DOCUMENT]"""
                        result = chatgpt_generation(prompt, review_text)
                        
                        st.success("✅ Analysis complete!")
                        st.markdown("**ChatGPT Analysis:**")
                        st.write(result)
                        
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
        else:
            st.warning("⚠️ Please enter a review before analyzing!")

# ============================================
# TAB 2: BATCH ANALYSIS
# ============================================
with tab2:
    st.markdown("### 📁 Upload your dataset")
    
    st.info("""
    **📋 File Requirements:**
    - Supported formats: CSV, Excel (.xlsx, .xls)
    - Must contain a column with text reviews
    - Maximum 1000 rows recommended for ChatGPT (API limits)
    """)
    
    uploaded_file = st.file_uploader(
        "Drag and drop your file here",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a CSV or Excel file containing movie reviews"
    )
    
    if uploaded_file is not None:
        try:
            # Leer el archivo
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File loaded successfully! {len(df)} rows found.")
            
            # Mostrar preview
            with st.expander("👀 Preview Data"):
                st.dataframe(df.head(10))
            
            # Seleccionar columna de texto
            st.markdown("### 🎯 Select the text column")
            text_column = st.selectbox(
                "Which column contains the reviews?",
                options=df.columns.tolist(),
                help="Select the column that contains the text to analyze"
            )
            
            # Mostrar muestra de la columna seleccionada
            st.markdown("**Sample from selected column:**")
            st.write(df[text_column].head(3).tolist())
            
            # Botón de análisis
            col1, col2 = st.columns([2, 1])
            with col1:
                analyze_batch = st.button(
                    f"🚀 Analyze {len(df)} reviews with {model_option}",
                    type="primary",
                    use_container_width=True
                )
            with col2:
                if model_option == "ChatGPT" and len(df) > 100:
                    st.warning(f"⚠️ {len(df)} rows may be expensive!")
            
            if analyze_batch:
                with st.spinner(f"Analyzing {len(df)} reviews... This may take a while."):
                    try:
                        # Realizar predicciones según el modelo
                        if model_option == "Pipeline Model (Transformer)":
                            model = laod_pipeline()
                            predictions_df = predict_batch_pipeline(df, text_column, model)
                        
                        elif model_option == "Logistic Regression":
                            lr_model = load_lr_model()
                            model = SentenceTransformer(MODEL_EMBEDDING)
                            predictions_df = predict_batch_lr(df, text_column, lr_model, model)
                        
                        elif model_option == "FLAN-T5 (Encoder-Decoder)":
                            model = load_encoder_decoder_model()
                            predictions_df = predict_batch_t5(df, text_column, model)
                        
                        else:  # ChatGPT
                            if len(df) > 100:
                                st.warning("⚠️ Processing large dataset with ChatGPT. This will take time and may incur costs.")
                            predictions_df = predict_batch_chatgpt(df, text_column, MODEL_NAME_CHATGPT)
                        
                        # Combinar resultados con el dataframe original
                        result_df = pd.concat([df, predictions_df], axis=1)
                        
                        st.success("✅ Batch analysis complete!")
                        
                        # Mostrar estadísticas
                        st.markdown("### 📊 Results Summary")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            positive_count = (predictions_df['Sentiment'] == 'Positive').sum()
                            st.metric("Positive Reviews", positive_count, 
                                     f"{positive_count/len(df)*100:.1f}%")
                        
                        with col2:
                            negative_count = (predictions_df['Sentiment'] == 'Negative').sum()
                            st.metric("Negative Reviews", negative_count,
                                     f"{negative_count/len(df)*100:.1f}%")
                        
                        with col3:
                            avg_confidence = predictions_df['Confidence'].mean()
                            st.metric("Avg Confidence", f"{avg_confidence:.2%}")
                        
                        # Mostrar resultados
                        st.markdown("### 📋 Detailed Results")
                        st.dataframe(result_df, use_container_width=True)
                        
                        # Botón de descarga
                        csv = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name=f"sentiment_analysis_results_{model_option.replace(' ', '_')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error during batch analysis: {str(e)}")
                        st.exception(e)
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("Please make sure your file is a valid CSV or Excel file.")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #888;'>Made with ❤️ using Streamlit | 🍅 TomatoMeter AI</p>",
    unsafe_allow_html=True
)