# Customer Feedback Classification with BERT

An NLP project that classifies customer feedback into Positive 😊 or Negative 😠 sentiment using a BERT encoder-only transformer. This system helps businesses automatically analyze customer sentiment in real time.

## Project Overview

Customer feedback is a goldmine of insights, but manually analyzing it is time-consuming. This project leverages BERT (bert-base-uncased) to automatically classify textual feedback into positive or negative sentiments.

Problem Statement

Perform sentiment classification on customer feedback to determine whether the sentiment is positive, negative, or neutral (neutral planned for future work).

Dataset:
Customer Feedback Dataset on Kaggle

Objective:
Fine-tune a BERT-based model to classify textual feedback according to sentiment.

##  Key Features

Fine-tuned BERT (encoder-only) for sentiment classification

Real-time web app deployed with Streamlit

Accurate results even with a small dataset

Provides metrics: Accuracy, F1-Score, Confusion Matrix

Example predictions to demonstrate performance

## Performance
Metric	Score
Accuracy	90%
F1-Score (Macro)	0.8958
Zero False Negatives on Positive Reviews	

Even with a limited dataset, BERT demonstrated excellent performance, showcasing the power of transfer learning for NLP.

-> Tech Stack

Language & Libraries: Python, PyTorch

NLP Framework: Hugging Face Transformers

Web App: Streamlit

-> Highlights

Real-time web interface: Test the model live with user inputs

Encoder-only architecture: Captures contextual information effectively

Explainable and robust: Minimal compute with highly accurate predictions

## Future Work

Add a Neutral class for more nuanced sentiment analysis

Integrate Explainable AI tools like LIME or SHAP

## Deliverables

Preprocessing & tokenization script

Training & validation pipeline

Evaluation metrics (accuracy, F1-score, confusion matrix)

Example predictions

 Try It Out

Streamlit App: https://sentimentalanalyzer-shraqjwcfrvy7bg9uec3z8.streamlit.app/

Medium Article: https://aneelabashir425.medium.com/encoder-only-bert-customer-feedback-classification-using-streamlit-1137a0b4d0c1
