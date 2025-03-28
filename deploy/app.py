import streamlit as st
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.system("pip install tensorflow-cpu==2.11.0")
import tensorflow
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer used in training
tokenizer = Tokenizer(num_words=10000)
# You must re-train or load tokenizer from a JSON if you saved it!
tokenizer.fit_on_texts(["dummy"])  # Temporary; replace with loaded tokenizer

# Preprocess text for models
def preprocess(text):
    sequence = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequence, maxlen=100)

# Load Keras models
model1 = load_model("model1.h5")  # Suicide risk
model2 = load_model("best_model (2).keras")  # Diagnosis classifier

# Model prediction wrappers
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

# Session memory
if "history" not in st.session_state:
    st.session_state.history = []

def therapist_pipeline(user_input):
    st.session_state.history.append(f"User: {user_input}")
    risk = model1_predict(user_input)

    if risk == 1:
        response = (
            "I'm really sorry you're feeling this way. You're not alone — please talk to someone you trust "
            "or a professional. I'm here to listen, but it's important to get real support too. Please contact 9-8-8 if you need immediate support. I hope you get better. 💙"
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

# Streamlit UI
st.title("🧠 TARS.help")
user_input = st.text_input("How are you feeling today?")

if user_input:
    response = therapist_pipeline(user_input)
    st.markdown(f"**AI Therapist:** {response}")

if st.button("🧾 Generate Therapist Summary"):
    st.markdown("### 🧠 Session Summary")
    st.markdown(summarize_session())

# Show history
for i in range(0, len(st.session_state.history), 2):
    st.markdown(f"**You:** {st.session_state.history[i][6:]}")
    st.markdown(f"**AI:** {st.session_state.history[i+1][4:]}")
