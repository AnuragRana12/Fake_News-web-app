#!/usr/bin/env python
# coding: utf-8

# ===============================================
# Fake News Detection Web App
# Streamlit + Logistic Regression/SVM + GNews Verification
# With Particle Background + Sticky Header + Glowing Top Menu
# ===============================================
# pip install streamlit scikit-learn pandas requests newspaper3k streamlit-option-menu

import pickle
import pandas as pd
import requests
import re
from pathlib import Path
from newspaper import Article
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
from streamlit.components.v1 import html
from streamlit_option_menu import option_menu

# -----------------------
# Utility: Text Cleaning
# -----------------------
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

# -----------------------
# 1. Train or Load Model
# -----------------------
model_dir = Path("fake_news_model")
model_file = model_dir / "model.pkl"
vec_file = model_dir / "vectorizer.pkl"

if not model_file.exists() or not vec_file.exists():
    st.info("Training model for the first time... please wait.")
    fake_path = r"D:\SS\archive (1)\Fake.csv"
    real_path = r"D:\SS\archive (1)\Real.csv"

    fake_df = pd.read_csv(fake_path)
    real_df = pd.read_csv(real_path)
    if "text" not in fake_df.columns or "text" not in real_df.columns:
        st.error("Your CSV files must contain at least a 'text' column.")
        st.stop()

    fake_df["content"] = (fake_df.get("title", "").astype(str) + " " + fake_df["text"].astype(str)).apply(clean_text)
    real_df["content"] = (real_df.get("title", "").astype(str) + " " + real_df["text"].astype(str)).apply(clean_text)
    fake_df["label"] = 1
    real_df["label"] = 0

    df = pd.concat([fake_df[["content","label"]], real_df[["content","label"]]], ignore_index=True).dropna()

    X_train, X_test, y_train, y_test = train_test_split(
        df["content"].values, df["label"].values, test_size=0.25, random_state=42, stratify=df["label"].values
    )

    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1,2), max_features=20000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=3000, C=2, class_weight='balanced')
    model.fit(X_train_tfidf, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_tfidf))

    if acc < 0.80:
        st.warning(f"Logistic Regression accuracy {acc*100:.2f}% below 80%, switching to SVM...")
        model = LinearSVC()
        model.fit(X_train_tfidf, y_train)
        acc = accuracy_score(y_test, model.predict(X_test_tfidf))
        st.success(f"SVM trained successfully! Accuracy: {acc*100:.2f}%")
    else:
        st.success(f"Logistic Regression trained successfully! Accuracy: {acc*100:.2f}%")

    model_dir.mkdir(exist_ok=True)
    with open(vec_file, "wb") as f: pickle.dump(vectorizer, f)
    with open(model_file, "wb") as f: pickle.dump(model, f)
else:
    with open(vec_file, "rb") as f: vectorizer = pickle.load(f)
    with open(model_file, "rb") as f: model = pickle.load(f)

# -----------------------
# 2. ML Prediction
# -----------------------
def predict_news(text):
    X_vec = vectorizer.transform([clean_text(text)])
    if hasattr(model,"predict_proba"):
        prob = model.predict_proba(X_vec)[0]
        pred = model.predict(X_vec)[0]
        return {"prediction":"Fake" if pred==1 else "Real","real_score":float(prob[0]),"fake_score":float(prob[1])}
    else:
        pred = model.predict(X_vec)[0]
        return {"prediction":"Fake" if pred==1 else "Real","real_score":1.0 if pred==0 else 0.0,"fake_score":1.0 if pred==1 else 0.0}

# -----------------------
# 3. Verify with GNews
# -----------------------
def verify_with_gnews(api_key, query):
    url = f"https://gnews.io/api/v4/search"
    params = {"q": query, "token": api_key, "lang": "en", "max": 3}
    try:
        response = requests.get(url, params=params).json()
        return "articles" in response and len(response["articles"])>0
    except:
        return False

SECOND_API_KEY = "c268d145437a5053a62247fd65677990"  # replace with your key

# -----------------------
# 4. Hybrid Prediction
# -----------------------
def hybrid_prediction(text):
    ml_result = predict_news(text)
    verified = verify_with_gnews(SECOND_API_KEY, text[:100])
    if verified and ml_result["prediction"]=="Real":
        conf = max(ml_result["real_score"],0.80)
    elif not verified and ml_result["prediction"]=="Real":
        conf = min(ml_result["real_score"],0.60)
    else:
        conf = ml_result["fake_score"] if ml_result["prediction"]=="Fake" else ml_result["real_score"]
    return {
        "prediction":ml_result["prediction"],
        "real_score": conf if ml_result["prediction"]=="Real" else 1-conf,
        "fake_score": conf if ml_result["prediction"]=="Fake" else 1-conf,
        "verified":verified
    }

# -----------------------
# 5. Fetch Latest News
# -----------------------
def fetch_latest_news(api_key,country="in",limit=10):
    url="https://newsapi.org/v2/top-headlines"
    params={"apiKey":api_key,"country":country,"pageSize":limit}
    response=requests.get(url,params=params).json()
    if response.get("status")!="ok" or len(response.get("articles",[]))==0:
        url="https://newsapi.org/v2/everything"
        params={"apiKey":api_key,"q":"news","language":"en","pageSize":limit,"sortBy":"publishedAt"}
        response=requests.get(url,params=params).json()
    if response.get("status")!="ok":
        st.error(f"News API error: {response.get('message','Unknown error')}")
        return []
    articles=[]
    for a in response.get("articles",[]):
        title=a.get("title","").strip()
        desc=a.get("description","").strip()
        img=a.get("urlToImage","")
        full=f"{title} {desc}".strip()
        if title or desc:
            articles.append({"headline":full,"image":img})
    return articles[:limit]

# -----------------------
# 6. Extract Text from URL
# -----------------------
def extract_text_from_url(url):
    try:
        art=Article(url)
        art.download()
        art.parse()
        return (art.title or "")+" "+(art.text or "")
    except:
        return url

# -----------------------
# Streamlit Page Config
# -----------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

# Particle background
particles_html = """
<script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
<div id="particles-js" style="position:fixed;width:100%;height:100%;z-index:-1;top:0;left:0;"></div>
<script>
particlesJS("particles-js",{
  "particles":{"number":{"value":80},"size":{"value":3},"move":{"speed":1.8},
  "line_linked":{"enable":true},"opacity":{"value":0.5},"color":{"value":"#00ffff"}}
});
</script>
"""
html(particles_html,height=0)

# Custom CSS
st.markdown("""
<style>
body, html, .stApp {background: transparent !important;}
h1 {
    font-size:3rem !important;font-weight:800 !important;
    background:linear-gradient(270deg,#ff00ff,#00ffff,#ffcc00);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    animation:float 3s ease-in-out infinite,gradient 6s ease infinite;
    text-align:center;margin-top:0;
}
@keyframes float {0%{transform:translateY(0);}50%{transform:translateY(-10px);}100%{transform:translateY(0);}}
@keyframes gradient {0%{background-position:0% 50%;}50%{background-position:100% 50%;}100%{background-position:0% 50%;}}
.stButton>button {
    background:linear-gradient(90deg,#00c6ff,#0072ff);
    color:white;font-weight:bold;border:none;border-radius:12px;
    padding:0.6em 1.2em;box-shadow:0 4px 15px rgba(0,0,0,0.3);
    transition:all 0.3s ease-in-out;
}
.stButton>button:hover {transform:scale(1.05);box-shadow:0 6px 20px rgba(0,0,0,0.45);
background:linear-gradient(90deg,#0072ff,#00c6ff);}
</style>
""",unsafe_allow_html=True)

# Title
st.markdown("<h1>📰 Fake News Detection</h1><p style='text-align:center;'>AI-powered detection with live verification</p>",unsafe_allow_html=True)

# Navigation menu (top bar)
menu = option_menu(
    None, ["Check News","Live Headlines"],
    icons=["search","newspaper"], menu_icon="cast", default_index=0,
    orientation="horizontal",
    styles={
        "container":{"padding":"0!important","background-color":"#111"},
        "icon":{"color":"white","font-size":"20px"},
        "nav-link":{
            "font-size":"18px","color":"white","padding":"10px","margin":"0 5px",
            "border-radius":"8px","background":"linear-gradient(90deg,#ff7e5f,#feb47b)",
            "box-shadow":"0px 4px 10px rgba(0,0,0,0.4)"
        },
        "nav-link-selected":{
            "background":"linear-gradient(90deg,#00c6ff,#0072ff)",
            "color":"white","box-shadow":"0px 6px 15px rgba(0,0,0,0.5)"
        }
    }
)

# -----------------------
# Menu Logic
# -----------------------
if menu=="Check News":
    st.subheader("🔍 Analyze Any News Content or URL")
    user_input=st.text_area("Paste news text or link here:",height=180,placeholder="Enter text or paste a link...")
    if st.button("Analyze News"):
        if user_input.strip():
            with st.spinner("Analyzing with AI and verifying with GNews..."):
                text = extract_text_from_url(user_input) if user_input.startswith("http") else user_input
                result = hybrid_prediction(text)
            bg = "linear-gradient(90deg,#2ecc71,#27ae60)" if result['prediction']=="Real" else "linear-gradient(90deg,#e74c3c,#c0392b)"
            icon = "✅" if result['prediction']=="Real" else "❌"
            verify_icon = "🔗" if result["verified"] else "⚠"
            verify_text = "Verified by trusted sources" if result["verified"] else ""
            st.markdown(f"""
            <div style='padding:20px;border-radius:15px;background:{bg};color:white;margin-top:15px;box-shadow:0 6px 20px rgba(0,0,0,0.25);'>
                <h3>{icon} Prediction: {result['prediction']} News {verify_icon}</h3>
                <p><b>Real Probability:</b> {result['real_score']*100:.1f}%<br>
                   <b>Fake Probability:</b> {result['fake_score']*100:.1f}%</p>
                <p>{verify_text}</p>
            </div>
            """,unsafe_allow_html=True)
        else:
            st.warning("Please enter some text or URL.")

elif menu=="Live Headlines":
    st.subheader("🗞 Live News Headlines (India)")
    API_KEY="cf9d1a74cf564c5daf851dca5b256769"
    with st.spinner("Fetching top headlines..."):
        articles=fetch_latest_news(API_KEY,country="in",limit=5)
    if not articles:
        st.warning("No headlines available at the moment.")
    else:
        for a in articles:
            result = hybrid_prediction(a["headline"])
            bg = "linear-gradient(90deg,#2ecc71,#27ae60)" if result['prediction']=="Real" else "linear-gradient(90deg,#e74c3c,#c0392b)"
            icon = "✅" if result['prediction']=="Real" else "❌"
            verify_icon = "🔗" if result["verified"] else "⚠"
            verify_text = "Verified by trusted sources" if result["verified"] else ""
            with st.expander(a["headline"][:120]+"..."):
                if a["image"]:
                    st.image(a["image"],width=400,caption="News Thumbnail")
                st.markdown(f"""
                <div style='padding:20px;border-radius:15px;background:{bg};color:white;margin-top:15px;box-shadow:0 6px 20px rgba(0,0,0,0.25);'>
                    <h4>{icon} {result['prediction']} News {verify_icon}</h4>
                    <p><b>Real Probability:</b> {result['real_score']*100:.1f}%<br>
                       <b>Fake Probability:</b> {result['fake_score']*100:.1f}%</p>
                    <p>{verify_text}</p>
                </div>
                """,unsafe_allow_html=True)
