import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Dummy tokenizer logic for basic preprocessing
def preprocess(text):
    tokens = text.lower().split()
    word_index = {w: i+1 for i, w in enumerate(set(tokens))}
    sequence = [[word_index.get(word, 0) for word in tokens]]
    return pad_sequences(sequence, maxlen=100)

# Load your trained Keras models
model1 = load_model("model1.h5")
model2 = load_model("model2.h5")

def model1_predict(text):
    pred = model1.predict(preprocess(text))[0][0]
    return int(pred > 0.5)

def model2_predict(text):
    pred = model2.predict(preprocess(text))[0]
    return int(np.argmax(pred))

diagnosis_labels = {
    1: "Anxiety",
    2: "Depression",
    3: "Bipolar disorder",
    4: "PTSD",
    5: "OCD",
    6: "ADHD",
    7: "General emotional distress"
}

@st.cache_resource
def load_llm():
    model_id = "tiiuae/falcon-7b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype="auto"
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

generator = load_llm()

if "history" not in st.session_state:
    st.session_state.history = []

def therapist_pipeline(user_input):
    st.session_state.history.append(f"User: {user_input}")
    risk = model1_predict(user_input)

    if risk == 1:
        response = (
            "I'm really sorry you're feeling this way. You're not alone — please talk to someone you trust "
            "or a professional. I'm here to listen, but it's important to get real support too. 💙"
        )
    else:
        diagnosis_code = model2_predict(user_input)
        diagnosis = diagnosis_labels.get(diagnosis_code, "General emotional distress")

        prompt = f"""You are an empathetic AI therapist. The user has been diagnosed with {diagnosis}. Respond supportively.

User: {user_input}
AI:"""

        response = generator(prompt, max_new_tokens=150, temperature=0.7)[0]["generated_text"]
        response = response.split("AI:")[-1].strip()

    st.session_state.history.append(f"AI: {response}")
    return response

def summarize_session():
    session_text = "\n".join(st.session_state.history)
    prompt = f"""Summarize the emotional state of the user based on the following conversation. Include emotional cues and possible diagnoses. Write it like a therapist note.

Conversation:
{session_text}

Summary:"""
    summary = generator(prompt, max_new_tokens=250, temperature=0.5)[0]["generated_text"]
    return summary.split("Summary:")[-1].strip()

st.title("🧠 TARS.help - AI Therapist")
user_input = st.text_input("How are you feeling today?")

if user_input:
    response = therapist_pipeline(user_input)
    st.markdown(f"**AI Therapist:** {response}")

if st.button("🧾 Generate Therapist Summary"):
    st.markdown("### 🧠 Session Summary")
    st.markdown(summarize_session())

for i in range(0, len(st.session_state.history), 2):
    st.markdown(f"**You:** {st.session_state.history[i][6:]}")
    st.markdown(f"**AI:** {st.session_state.history[i+1][4:]}")
