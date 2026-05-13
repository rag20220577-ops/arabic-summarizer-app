# 📝 Arabic Text Summarizer

An end-to-end Arabic text summarization system using a **Seq2Seq Bidirectional LSTM model with Bahdanau Attention** and **Beam Search decoding**, built with PyTorch and deployed via Streamlit.

---

## 🗂️ Project Structure
arabic-summarizer-app
├── finalllnlp.ipynb        
├── app.py                  
└── requirements.txt  

---

## 📊 Dataset

The model was trained on a merged Arabic news dataset from 4 sources:
- **ArabSum** (JSONL format) — high quality article/summary pairs
- **Parquet dataset** — additional Arabic news pairs
- **Kaggle Arabic News Corpus** — raw Arabic news articles
- **AbsArSumCorpus DW** — Deutsche Welle Arabic article/lead pairs

---

## 🧠 Model Architecture

- **Encoder**: Bidirectional LSTM — reads the article forward and backward
- **Attention**: Bahdanau (Additive) Attention — focuses on relevant parts of the article at each decoding step
- **Decoder**: LSTM — generates the summary word by word
- **Decoding**: Beam Search (beam width = 6) for better quality output

---

## ⚙️ Training Details

| Parameter | Value |
|---|---|
| Vocabulary size | 20,000 |
| Embedding dimension | 64 |
| LSTM hidden size | 128 |
| Batch size | 16 |
| Optimizer | Adam |
| Loss function | Sparse Categorical Crossentropy |
| Max article length | 100 tokens |
| Max summary length | 30 tokens |
| Early stopping patience | 3 epochs |

---

## 📈 Evaluation (ROUGE Scores)

Evaluated on the test set using a custom Arabic ROUGE tokenizer:

| Metric | Score |
|---|---|
| ROUGE-1 | measured on test set |
| ROUGE-2 | measured on test set |
| ROUGE-L | measured on test set |

---

## 🖥️ Streamlit App (Phase 3)

### How to Run

**1. Clone the repo**
```bash
git clone https://github.com/rag20220577-ops/arabic-summarizer-app.git
cd arabic-summarizer-app
```

**2. Download model files and place them in the folder**
- [`tokenizer.pkl`](YOUR_GOOGLE_DRIVE_LINK)
- [`arabic_summarizer_model.h5`](YOUR_GOOGLE_DRIVE_LINK)
- [`model_config.json`](YOUR_GOOGLE_DRIVE_LINK)

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
streamlit run app.py
```

**5. Open your browser at:**
http://localhost:8501

---

## 🌟 App Features

- Paste any Arabic news article and get an automatic summary
- Adjustable Beam Width slider (1–5)
- Optional ROUGE scoring against a reference summary
- Diacritics (tashkeel) are automatically removed before processing

---

## ⚠️ Limitations

- The model is trained on Arabic **news articles** only
- Short or casual Arabic text will produce irrelevant summaries
- Best results with articles of at least 3–4 sentences
