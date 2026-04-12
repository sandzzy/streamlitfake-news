import streamlit as st

def show_header():
    st.title("🐦 Fake News Detection on Twitter")
    st.write("Analyze tweets and detect fake news")

def show_result(prediction, confidence):
    if prediction == 1:
        st.success(f"✅ Real Tweet ({confidence*100:.2f}%)")
    else:
        st.error(f"❌ Fake Tweet ({confidence*100:.2f}%)")
