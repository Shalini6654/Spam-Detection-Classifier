import streamlit as st
import pickle

# -----------------------------
# Load model and vectorizer
# -----------------------------
@st.cache_resource
def load_artifacts():
    with open("NB_Spam_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_artifacts()

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Spam Detection App", layout="centered")

st.title("📩 Spam Detection")
st.write("Enter a message and check whether it's **Spam** or **Not Spam**.")

user_input = st.text_area(
    "Message",
    height=150,
    placeholder="Type or paste your message here..."
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Message cannot be empty.")
    else:
        transformed_text = vectorizer.transform([user_input])
        prediction = model.predict_proba(transformed_text)[0][1]

        if prediction > 0.25:
            st.error("🚨 This message is **SPAM**")
        else:
            st.success("✅ This message is **NOT SPAM**")



