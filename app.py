import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# =======================================
# Load model and tokenizer from Hugging Face
# =======================================
@st.cache_resource
def load_model():
    model_name = "aneelaBashir22f3414/sentiment_model"  # Hugging Face repo path

    # Force re-download by disabling cache use
    tokenizer = BertTokenizer.from_pretrained(
        model_name,
        local_files_only=False,
        revision=None
    )
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        local_files_only=False,
        revision=None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device


model, tokenizer, device = load_model()

# =======================================
# Streamlit UI
# =======================================
st.set_page_config(page_title="Sentiment Analysis App", page_icon="💬", layout="centered")

st.title("💬 Sentiment Analysis with BERT")
st.write("Type a sentence below to analyze its sentiment (Positive or Negative).")

# ---------------------------------------
# Text input
# ---------------------------------------
user_input = st.text_area("Enter your text here:", placeholder="e.g. I absolutely love this product!")

# ---------------------------------------
# Analyze button
# ---------------------------------------
if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text first.")
    else:
        # Tokenize input
        inputs = tokenizer(
            user_input,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Predict sentiment
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()

        # Label mapping
        label = "😊 Positive" if pred == 1 else "😠 Negative"

        # Display results
        st.markdown(f"### Prediction: **{label}**")
        st.progress(confidence)
        st.caption(f"Confidence: {confidence:.2%}")

# =======================================
# Footer
# =======================================
st.markdown("---")
st.caption("Built with ❤️ using BERT and Streamlit")
