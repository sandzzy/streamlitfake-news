import streamlit as st
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------------------
# Sample Twitter Data (Training)
# -------------------------------
tweets = [
    "Government announces new policy",
    "Official news released today",
    "Click here to win money now",
    "Fake celebrity scandal click now",
    "Ministry confirms update",
    "You won lottery claim prize"
]

labels = [1, 1, 0, 0, 1, 0]

# -------------------------------
# Preprocessing (as per report)
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    return text

cleaned = [clean_text(t) for t in tweets]

# -------------------------------
# Feature Extraction (TF-IDF)
# -------------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned)

# -------------------------------
# Model Training (Logistic Regression)
# -------------------------------
model = LogisticRegression()
model.fit(X, labels)

# -------------------------------
# UI (User Input Module)
# -------------------------------
st.title("🐦 Fake News Detection on Twitter")

st.write("Enter a tweet to analyze whether it is REAL or FAKE")

user_input = st.text_area("Enter Tweet")

# -------------------------------
# Prediction + Output Module
# -------------------------------
if st.button("Analyze Tweet"):
    if user_input.strip() != "":
        cleaned_input = clean_text(user_input)
        vector = vectorizer.transform([cleaned_input])

        prediction = model.predict(vector)
        prob = model.predict_proba(vector)

        confidence = max(prob[0]) * 100

        # Output Result
        if prediction[0] == 1:
            st.success(f"✅ Real Tweet (Confidence: {confidence:.2f}%)")
        else:
            st.error(f"❌ Fake Tweet (Confidence: {confidence:.2f}%)")

        # -------------------------------
        # EXTRA FEATURES (FOR REPORT)
        # -------------------------------

        st.subheader("🔍 Analysis")

        # Tweet Length
        st.write(f"Tweet Length: {len(user_input.split())} words")

        # Red flag keywords
        fake_keywords = ["click", "win", "free", "lottery", "shocking"]

        detected_flags = [word for word in fake_keywords if word in cleaned_input]

        if detected_flags:
            st.warning(f"⚠️ Suspicious Words Detected: {', '.join(detected_flags)}")
        else:
            st.info("No suspicious keywords detected")

        # Simple reasoning
        st.subheader("🧠 Reasoning")

        if prediction[0] == 0:
            st.write("This tweet contains clickbait or promotional patterns commonly seen in fake news.")
        else:
            st.write("This tweet appears formal and informational, which is typical of real news.")

    else:
        st.warning("Please enter a tweet")
