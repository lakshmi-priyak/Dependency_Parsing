import streamlit as st
import spacy # type: ignore
import os
from spacy.tokens import Doc # type: ignore
from sklearn.metrics import classification_report

# Define the absolute path to the model
MODEL_PATH = os.path.join(os.getcwd(), "dependency_parser_model")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}. Please train it first.")
        st.stop()
    return spacy.load(MODEL_PATH)

# Load model
nlp = load_model()

# Streamlit UI
st.title("Dependency Parser using spaCy")
st.write("Enter a sentence to analyze its dependency parsing.")

# Text input
text_input = st.text_input("Enter your sentence:", "He studies AI.")

if st.button("Analyze"):
    doc = nlp(text_input)
    
    # Display dependency parsing results
    st.subheader("Dependency Parsing Results:")
    result = []
    for token in doc:
        result.append(f"{token.text} --> {token.dep_} (Head: {doc[token.head.i].text})")
    st.write("\n".join(result))
    
    # Evaluation
    true_labels = ["nsubj", "ROOT", "dobj", "punct"]
    pred_labels = [token.dep_ for token in doc]
    
    st.subheader("Evaluation Metrics:")
    try:
        report = classification_report(true_labels, pred_labels, zero_division=0, output_dict=True)
        st.json(report)
    except Exception as e:
        st.error(f"Error in evaluation: {str(e)}")
