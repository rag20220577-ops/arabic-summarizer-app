import streamlit as st
import pickle
import numpy as np
import json
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from rouge_score import rouge_scorer

# ─── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Arabic Text Summarizer",
    page_icon="📝",
    layout="centered"
)

# ─── Load Model & Tokenizer ───────────────────────────────────
@st.cache_resource
def load_assets():
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    model = load_model("arabic_summarizer_model.h5")
    with open("model_config.json", "r") as f:
        config = json.load(f)
    return tokenizer, model, config

tokenizer, model, config = load_assets()

max_article_len = config["max_article_len"]
max_summary_len = config["max_summary_len"]
reverse_word_index = tokenizer.index_word
target_word_index  = tokenizer.word_index

# ─── Remove Diacritics ────────────────────────────────────────
def remove_diacritics(text):
    diacritics = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    return diacritics.sub('', text)

# ─── Beam Search ──────────────────────────────────────────────
def beam_search_decode(input_seq, beam_width=3):
    start_token = target_word_index['sostoken']
    end_token   = target_word_index['eostoken']
    sequences   = [([start_token], 0.0)]

    for _ in range(max_summary_len):
        all_candidates = []
        for seq, score in sequences:
            target_seq    = np.array(seq).reshape(1, -1)
            output_tokens = model.predict([input_seq, target_seq], verbose=0)
            probs         = output_tokens[0, -1, :]
            top_indices   = np.argsort(probs)[-beam_width:][::-1]

            for idx in top_indices:
                word = reverse_word_index.get(idx, "")
                if not word or (len(seq) > 1 and idx == seq[-1]):
                    continue
                new_seq   = seq + [idx]
                new_score = score + np.log(probs[idx] + 1e-9)
                all_candidates.append((new_seq, new_score))

        if not all_candidates:
            break
        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        if any(seq[-1] == end_token for seq, _ in sequences):
            break

    best_seq = sequences[0][0]
    words = [
        reverse_word_index.get(idx, "")
        for idx in best_seq
        if reverse_word_index.get(idx, "") not in ["sostoken", "eostoken", ""]
    ]
    return " ".join(words)

def summarize(text: str, beam_width: int) -> str:
    text   = remove_diacritics(text)
    seq    = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_article_len, padding='post')
    return beam_search_decode(padded, beam_width=beam_width)

# ─── ROUGE Tokenizer ──────────────────────────────────────────
class ArabicTokenizer:
    def tokenize(self, text):
        text = re.sub(r'[^\u0600-\u06FF\s]', ' ', str(text))
        return re.sub(r'\s+', ' ', text).strip().split()

# ─── UI ───────────────────────────────────────────────────────
st.title("📝 Arabic Text Summarizer")
st.markdown("Paste any Arabic article below and get an automatic summary.")

user_input = st.text_area(
    "Arabic Article",
    height=250,
    placeholder="اكتب أو الصق النص العربي هنا..."
)

beam_width = st.slider("Beam Width (higher = better but slower)", 1, 5, 3)

summary = ""

if st.button("Generate Summary", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Generating summary..."):
            summary = summarize(user_input, beam_width)
        st.subheader("Summary")
        st.success(summary)
        st.caption(f"Input length: {len(user_input.split())} words | Summary length: {len(summary.split())} words")

