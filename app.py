import streamlit as st
import pandas as pd
import pickle
import numpy as np
import re

@st.cache_resource
def load_models():
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("tfidf_lr_model.pkl", "rb") as f:
        star_model = pickle.load(f)
    with open("sentiment_model.pkl", "rb") as f:
        sentiment_model = pickle.load(f)
    return vectorizer, star_model, sentiment_model

@st.cache_data
def load_data():
    return pd.read_csv("data_clean.csv")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

vectorizer, star_model, sentiment_model = load_models()
data = load_data()

st.title("Insurance Review Analyzer")
page = st.sidebar.selectbox("Choose a page", ["Prediction", "Summary", "Explanation", "Information Retrieval", "RAG", "QA"])

if page == "Prediction":
    st.header("Predict Star Rating & Sentiment")
    user_input = st.text_area("Enter your review:", height=150)
    if st.button("Predict"):
        if user_input:
            cleaned = clean_text(user_input)
            tfidf = vectorizer.transform([cleaned])
            star = star_model.predict(tfidf)[0]
            sentiment = sentiment_model.predict(tfidf)[0]
            st.success(f"Predicted Stars: {int(star)} / 5")
            st.info(f"Sentiment: {sentiment}")
        else:
            st.warning("Please enter a review.")

elif page == "Summary":
    st.header("Review Summary by Insurer")
    insurer = st.selectbox("Select an insurer:", data['assureur'].unique())
    insurer_data = data[data['assureur'] == insurer]
    st.write(f"Total reviews: {len(insurer_data)}")
    st.write(f"Average rating: {insurer_data['note'].mean():.2f} / 5")
    st.subheader("Recent reviews summary:")
    samples = insurer_data['avis_summary'].dropna().head(5).tolist()
    for i, s in enumerate(samples):
        st.write(f"{i+1}. {s}")
    st.subheader("Rating distribution:")
    st.bar_chart(insurer_data['note'].value_counts().sort_index())

elif page == "Explanation":
    st.header("Prediction Explanation")
    user_input = st.text_area("Enter your review:", height=150)
    if st.button("Explain"):
        if user_input:
            cleaned = clean_text(user_input)
            tfidf = vectorizer.transform([cleaned])
            star = star_model.predict(tfidf)[0]
            sentiment = sentiment_model.predict(tfidf)[0]
            st.success(f"Predicted Stars: {int(star)} / 5")
            st.info(f"Sentiment: {sentiment}")
            feature_names = vectorizer.get_feature_names_out()
            tfidf_array = tfidf.toarray()[0]
            top_indices = tfidf_array.argsort()[::-1][:10]
            top_words = [(feature_names[i], tfidf_array[i]) for i in top_indices if tfidf_array[i] > 0]
            st.subheader("Most influential words:")
            for word, score in top_words:
                st.write(f"**{word}**: {score:.4f}")
        else:
            st.warning("Please enter a review.")

elif page == "Information Retrieval":
    st.header("Search Reviews")
    query = st.text_input("Enter keywords to search:")
    star_filter = st.multiselect("Filter by stars:", [1, 2, 3, 4, 5], default=[1,2,3,4,5])
    if st.button("Search"):
        if query:
            filtered = data[data['note'].isin(star_filter)]
            results = filtered[filtered['avis'].str.contains(query, case=False, na=False)]
            st.write(f"Found {len(results)} reviews")
            for _, row in results.head(10).iterrows():
                st.write(f"**{int(row['note'])} stars** | {row['assureur']}")
                st.write(row['avis'][:200])
                st.divider()
        else:
            st.warning("Please enter a keyword.")

elif page == "RAG":
    st.header("RAG - Ask about Insurers")
    query = st.text_input("Ask a question about insurance reviews:")
    if st.button("Search & Answer"):
        if query:
            results = data[data['avis'].str.contains(query, case=False, na=False)]
            context = " ".join(results['avis_en'].dropna().head(5).tolist())
            st.subheader("Relevant reviews found:")
            for _, row in results.head(3).iterrows():
                st.write(f"**{int(row['note'])} stars** | {row['assureur']}")
                st.write(row['avis'][:200])
                st.divider()
            st.subheader("Context summary:")
            st.write(context[:500])
        else:
            st.warning("Please enter a question.")

elif page == "QA":
    st.header("Q&A about Insurers")
    insurer = st.selectbox("Select an insurer:", data['assureur'].unique())
    question = st.text_input("Ask a question:")
    if st.button("Answer"):
        if question:
            insurer_data = data[data['assureur'] == insurer]
            avg_rating = insurer_data['note'].mean()
            total = len(insurer_data)
            positive = len(insurer_data[insurer_data['note'] >= 4])
            negative = len(insurer_data[insurer_data['note'] <= 2])
            st.write(f"**Insurer:** {insurer}")
            st.write(f"**Total reviews:** {total}")
            st.write(f"**Average rating:** {avg_rating:.2f} / 5")
            st.write(f"**Positive reviews (4-5 stars):** {positive} ({positive/total*100:.1f}%)")
            st.write(f"**Negative reviews (1-2 stars):** {negative} ({negative/total*100:.1f}%)")
            st.subheader("Sample reviews:")
            samples = insurer_data.sample(min(3, len(insurer_data)))
            for _, row in samples.iterrows():
                st.write(f"**{int(row['note'])} stars:** {row['avis'][:150]}")
                st.divider()
        else:
            st.warning("Please enter a question.")
