import streamlit as st
import pickle

# Load the saved model and vectorizer
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit app
st.title("Email Spam Classifier")
st.subheader("Determine whether an email is spam or ham")

# Input text
input_text = st.text_area("Enter the email text here:")

if st.button("Classify"):
    if input_text:
        # Preprocess and predict
        transformed_text = vectorizer.transform([input_text])
        prediction = model.predict(transformed_text)[0]
        result = "Spam" if prediction == 1 else "Ham"
        st.success(f"The email is classified as: {result}")
    else:
        st.warning("Please enter some text to classify.")
