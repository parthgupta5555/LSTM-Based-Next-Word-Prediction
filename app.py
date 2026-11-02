import html
import pickle
from pathlib import Path

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="💬",
    layout="centered",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "lstm_model.h5"
TOKENIZER_PATH = BASE_DIR / "tokenizer.pkl"
MAX_LEN_PATH = BASE_DIR / "max_len.pkl"


# ---------------------------------------------------------
# Load the trained model and tokenizer only once
# ---------------------------------------------------------
@st.cache_resource
def load_assets():
    missing_files = [
        path.name
        for path in (MODEL_PATH, TOKENIZER_PATH, MAX_LEN_PATH)
        if not path.exists()
    ]
    if missing_files:
        raise FileNotFoundError("Missing file(s): " + ", ".join(missing_files))

    # Only load pickle files that you created or trust.
    with TOKENIZER_PATH.open("rb") as file:
        tokenizer = pickle.load(file)
    with MAX_LEN_PATH.open("rb") as file:
        max_len = int(pickle.load(file))

    # compile=False avoids needing the original optimizer/loss for inference.
    model = load_model(MODEL_PATH, compile=False)
    index_to_word = {
        index: word for word, index in tokenizer.word_index.items()
    }
    return model, tokenizer, max_len, index_to_word


# ---------------------------------------------------------
# Predict one next word
# ---------------------------------------------------------
def predict_next_word(model, tokenizer, index_to_word, text, max_len):
    sequence = tokenizer.texts_to_sequences([text])[0]
    if not sequence:
        return None, "I couldn't recognize the words in that input. Try another phrase."

    padded = pad_sequences(
        [sequence], maxlen=max_len, padding="pre", truncating="pre"
    )
    prediction = np.asarray(model.predict(padded, verbose=0))

    if prediction.ndim != 2 or prediction.shape[0] != 1:
        return None, "The model returned an unexpected output shape."

    # Ignore padding (index 0) and output indices absent from this tokenizer.
    probabilities = prediction[0].copy()
    valid_indices = [
        index for index in index_to_word if 0 < index < len(probabilities)
    ]
    if not valid_indices:
        return None, "No valid vocabulary indices were found in the model output."

    valid_mask = np.zeros(len(probabilities), dtype=bool)
    valid_mask[valid_indices] = True
    probabilities[~valid_mask] = -np.inf

    predicted_index = int(np.argmax(probabilities))
    predicted_word = index_to_word.get(predicted_index)
    if predicted_word is None:
        return None, "The prediction could not be matched to a vocabulary word."
    return predicted_word, None


# ---------------------------------------------------------
# Generate one or more words
# ---------------------------------------------------------
def generate_text(model, tokenizer, index_to_word, text, max_len, word_count):
    current_text = text.strip()
    for _ in range(word_count):
        next_word, error = predict_next_word(
            model, tokenizer, index_to_word, current_text, max_len
        )
        if error:
            return None, error
        current_text += " " + next_word
    return current_text, None


# ---------------------------------------------------------
# User interface
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 2.35rem;
        font-weight: 750;
        margin-bottom: 0.25rem;
    }
    .subtitle {
        text-align: center;
        color: #888888;
        margin-bottom: 1.8rem;
    }
    .result-card {
        padding: 1.25rem;
        border: 1px solid rgba(128, 128, 128, 0.35);
        border-radius: 14px;
        margin-top: 0.75rem;
        line-height: 1.7;
        overflow-wrap: anywhere;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">💬 Next Word Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Enter a sentence and let the LSTM predict what comes next.</div>',
    unsafe_allow_html=True,
)

try:
    model, tokenizer, max_len, index_to_word = load_assets()
except Exception as error:
    st.error(f"Unable to load the model files: {error}")
    st.info(
        "Keep app.py, lstm_model.h5, tokenizer.pkl, and max_len.pkl "
        "together in the same project folder."
    )
    st.stop()

user_text = st.text_area(
    "Enter your text",
    placeholder="For example: What are you",
    height=110,
)

word_count = st.slider(
    "Number of words to generate",
    min_value=1,
    max_value=10,
    value=3,
)

predict_col, clear_col = st.columns(2)
with predict_col:
    predict_clicked = st.button(
        "Predict Next Words", type="primary", use_container_width=True
    )
with clear_col:
    clear_clicked = st.button("Clear", use_container_width=True)

if clear_clicked:
    st.rerun()

if predict_clicked:
    if not user_text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Generating text..."):
            result, error = generate_text(
                model, tokenizer, index_to_word, user_text, max_len, word_count
            )
        if error:
            st.warning(error)
        else:
            st.markdown("### Generated text")
            st.markdown(
                f'<div class="result-card">{html.escape(result)}</div>',
                unsafe_allow_html=True,
            )
