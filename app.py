import streamlit as st
import pandas as pd
import pickle
import torch
import torch.nn as nn
import re
from transformers import pipeline


# -----------------------------
# BiLSTM architecture
# -----------------------------
class BiLSTM(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, 64, padding_idx=0
        )

        self.lstm = nn.LSTM(
            64, 64,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(128, 1)

    def forward(self, x):

        x = self.embedding(x)

        _, (hidden, _) = self.lstm(x)

        x = torch.cat(
            (hidden[-2], hidden[-1]),
            dim=1
        )

        x = self.dropout(x)

        return torch.sigmoid(
            self.fc(x)
        ).squeeze(1)


# -----------------------------
# Load saved BiLSTM .pth
# -----------------------------
checkpoint = torch.load(
    "bilstm_urgency.pth",
    map_location="cpu"
)

vocab = checkpoint["vocab"]
MAX_LEN = checkpoint["max_len"]

bilstm_model = BiLSTM(len(vocab))

bilstm_model.load_state_dict(
    checkpoint["model_state_dict"]
)

bilstm_model.eval()


# -----------------------------
# BiLSTM urgency prediction
# -----------------------------
def predict_urgency(text):

    words = re.sub(
        r"[^a-zA-Z0-9\s]",
        "",
        text.lower()
    ).split()

    sequence = [
        vocab.get(word, 1)
        for word in words
    ][:MAX_LEN]

    sequence += [0] * (
        MAX_LEN - len(sequence)
    )

    x = torch.tensor([sequence])

    with torch.no_grad():
        urgency = bilstm_model(x).item()

    return urgency


# -----------------------------
# Load saved files
# -----------------------------
df = pd.read_csv(
    "omnifeedback_raw_dataset.csv"
)

with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("random_forest.pkl", "rb") as f:
    rf_model = pickle.load(f)


# -----------------------------
# BERT NER
# -----------------------------
ner = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"
)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="OmniFeedback AI"
)

st.title("OmniFeedback AI")
st.write(
    "Enterprise Customer Feedback "
    "Analytics & Intelligence Copilot"
)


page = st.sidebar.radio(
    "Menu",
    [
        "Home",
        "EDA",
        "Feedback Copilot"
    ]
)


# =============================
# HOME
# =============================
if page == "Home":

    st.header("Customer Feedback Intelligence")

    st.write(
        "Analyze customer feedback using "
        "Machine Learning, BiLSTM and NLP."
    )


# =============================
# EDA
# =============================
elif page == "EDA":

    st.header("Feedback Analytics")

    st.subheader("Urgency Score")

    st.bar_chart(
        df["urgency_score"]
    )

    st.subheader("Feedback by Channel")

    st.bar_chart(
        df["channel"].value_counts()
    )

    st.subheader("Feedback by Label")

    st.bar_chart(
        df["label"].value_counts()
    )

    st.subheader("Aspect Category")

    st.bar_chart(
        df["aspect_category"].value_counts()
    )


# =============================
# FEEDBACK COPILOT
# =============================
else:

    st.header("Feedback Copilot")

    feedback = st.text_area(
        "Enter Customer Feedback"
    )

    if st.button("Analyze Feedback"):

        if feedback.strip() == "":
            st.warning(
                "Please enter customer feedback."
            )

        else:

            # Random Forest classification
            X = tfidf.transform(
                [feedback]
            )

            prediction = rf_model.predict(X)[0]

            # BiLSTM urgency prediction
            urgency = predict_urgency(
                feedback
            )

            # BERT entities
            entities = ner(feedback)

            entity_list = [
                {
                    "entity": e["word"],
                    "type": e["entity_group"]
                }
                for e in entities
            ]

            # Priority
            if urgency >= 0.7:
                priority = "High"
                action = (
                    "Escalate to support team immediately"
                )

            elif urgency >= 0.4:
                priority = "Medium"
                action = (
                    "Review and assign to support team"
                )

            else:
                priority = "Low"
                action = "Monitor feedback"


            # Results
            st.subheader("Prediction")

            st.write(
                "Classification:",
                prediction
            )

            st.write(
                "Urgency Score:",
                round(urgency, 2)
            )

            st.write(
                "Priority:",
                priority
            )

            st.write(
                "Recommended Action:",
                action
            )

            st.subheader(
                "Detected Entities"
            )

            st.write(entity_list)

            st.subheader(
                "Feedback"
            )

            st.write(feedback)