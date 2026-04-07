# Insurance Customer Review Analysis - NLP Project 2

## Project Overview
A complete NLP pipeline for analyzing French insurance customer reviews from 35 companies (34,435 reviews). The project covers data cleaning, topic modeling, word embeddings, supervised learning, and an interactive Streamlit application.

**Author:** Han Wang - ESILV NLP 2026

## Results
- Star rating prediction (5-class): **52.9%** accuracy (TF-IDF + Logistic Regression)
- Sentiment analysis (3-class): **81.5%** accuracy
- Key insight: Cancellation (1.47★) and claims processing (1.70★) are the most problematic areas

## Project Structure
- `nlp(2).ipynb` — Main notebook with full pipeline
- `app.py` — Streamlit application
- `data_clean.csv` — Cleaned dataset
- `word2vec.model` — Trained Word2Vec model
- `tfidf_vectorizer.pkl` — Saved TF-IDF vectorizer
- `tfidf_lr_model.pkl` — Saved Logistic Regression model
- `sentiment_model.pkl` — Saved sentiment model

## Pipeline Steps
1. **Data Cleaning** — text preprocessing, stopword removal, spelling correction (pyspellchecker)
2. **Summary Generation** — facebook/bart-large-cnn for summarization
3. **Topic Modeling** — LDA with 10 topics (Gensim), pyLDAvis visualization
4. **Word Embeddings** — Word2Vec (trained), GloVe (pre-trained), TensorBoard visualization, semantic search
5. **Supervised Learning** — TF-IDF + LR/SVM/NB, Embedding, CNN, zero-shot LLM
6. **Sentiment Analysis** — 3-class (negative/neutral/positive), 81.5% accuracy
7. **Streamlit App** — 6 interactive pages: Prediction, Summary, Explanation, Information Retrieval, RAG, QA

## Installation
```bash
pip install -r requirements.txt
```

## Run the Streamlit App
```bash
streamlit run app.py
```
