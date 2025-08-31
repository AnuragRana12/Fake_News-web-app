# 📰 Fake News Detection Web App

A Streamlit-based machine learning web application that detects whether a news article is **Real** or **Fake**.  
The app uses a Logistic Regression (with SVM fallback) model and also verifies articles using Google News for cross-checking.

---

## 🚀 Features
- **Text-based Fake News Detection** using a trained ML model  
- **Live News Verification** via Google News API  
- **Interactive Web UI** built with [Streamlit](https://streamlit.io)  
- **Custom animations, gradient UI, and particle effects**  
- **100% free to deploy on Streamlit Cloud**

---

## 📂 Project Structure
Fake_News-web-app/
│
├── app.py # Main Streamlit app file
├── model.pkl # Trained ML model
├── requirements.txt # Python dependencies
└── README.md # Project documentation
Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/AnuragRana12/Fake_News-web-app.git
cd Fake_News-web-app
2. Install dependencies
pip install -r requirements.txt
3. Run the Streamlit app
streamlit run app.py
