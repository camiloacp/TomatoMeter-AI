import streamlit as st

def predtiction_style(prediction):
    if prediction == 0:
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