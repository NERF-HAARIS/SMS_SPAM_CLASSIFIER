import download_nltk_data   # <- This will trigger NLTK data download on deploy
import streamlit as st
import pickle
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
import base64


# Set page config as the first command
st.set_page_config(page_title="Spam Message Classifier", page_icon="📱", layout="wide")

nltk.data.path.append('nltk_data')

ps = PorterStemmer()


# Function to set background image
# Set Background Image
def set_background(image_file, opacity=1):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        opacity: {opacity};
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background('spamback1.png', opacity=1)


# Text preprocessing
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum() and i not in stopwords.words('english') and i not in string.punctuation:
            y.append(ps.stem(i))

    return " ".join(y)


# Load model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Title
st.title("SPAM MESSAGE CLASSIFIER")

# Input message
input_msg = st.text_area("Enter the message")

# Prediction
if st.button("Predict"):
    transformed_msg = transform_text(input_msg)  # Preprocess the input
    vectorized_msg = tfidf.transform([transformed_msg])  # Vectorize
    result = model.predict(vectorized_msg)[0]  # Predict

    # Display result
    if result == 1:
        st.header("SPAM")
    else:
        st.header("NOT SPAM")
