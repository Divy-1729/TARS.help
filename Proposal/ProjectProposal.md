# 🧠 Your AI-Powered Assistant for Mental Health Professionals

## 🤔 What’s the Problem?

Mental health care in Canada is overwhelmed: long wait times, high costs, and the persistent stigma of seeking help leave millions underserved. In 2022, over 5 million Canadians aged 15+ experienced significant symptoms of mental illness. Generalized anxiety tripled from **3.8%** in 2012 to **11.9%**, while major depressive episodes doubled to **18.4%** [[1]](#1).

Wait times often exceed **six weeks**, delaying critical support. In Toronto, therapy costs between **$100–$300/hour**, making it inaccessible for many—especially uninsured populations like international students [[2]](#2). On top of that, **over 40%** of Canadians fear judgment for seeking help [[3]](#3), reinforcing silence over healing.

---

## 💬 Why This Matters

Mental health symptoms are often invisible—until they're not. We’ve seen it in the friend who stops responding, the student who suddenly skips lectures, the classmate whose laughter slowly fades. These aren’t isolated incidents. They're signals.

The system is too slow, too expensive, and too intimidating for many. But technology gives us a chance to intervene earlier, with care, empathy, and privacy. With AI, we can assist therapists in offering support that’s accessible, judgment-free, and instant.

---

## 🤖 Introducing the Assistant

This AI-powered assistant is designed to:

- **Chat with users** about how they’re feeling
- **Analyze sentiment** and emotional indicators
- **Summarize key insights** and potential diagnostic cues
- **Flag at-risk individuals** for escalation
- **Help therapists prepare** before sessions

### Powered by State-of-the-Art NLP

- **Bidirectional LSTM**: A recurrent neural network that captures long-range emotional patterns in both past and future context, making it ideal for understanding complex, sequential mental health signals [[8]](#8).
- **GPT-based response generation**: Delivers empathetic, context-aware replies to keep users engaged and validated.
- **Lightweight safety classifier**: Filters out harmful or inappropriate responses from the GPT engine, keeping interaction safe and clinically appropriate.

---

## 📊 Datasets That Make It Possible

To ensure the assistant is reliable and sensitive to mental health contexts, we’ve selected high-quality public datasets:

1. **Sentiment Analysis for Mental Health**  
   - 51,074 samples
   - Tags mental health-related text with emotional labels  
   [[4]](#4)

2. **Suicidal Mental Health Dataset**  
   - 20,000 samples labeled “Depression” or “Suicide Watch”  
   - Supports urgent risk detection and triage  
   [[5]](#5)

3. **Mental Disorders Reddit Dataset**  
   - 543,060 samples  
   - Labeled with: Bipolar, Depression, BPD, Anxiety, Schizophrenia  
   [[6]](#6)

4. **Mental Health NLP Conversations**  
   - ~2,500 therapist-client dialogues  
   - Used to fine-tune natural, contextual response generation  
   [[7]](#7)

---

## 🛠️ Core Functionality

| Feature                        | Description                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| 🧠 Sentiment Detection         | Real-time analysis of user messages using BiLSTM                            |
| 💬 Empathetic Responses        | GPT-based, context-aware replies that validate and engage                  |
| ⚠️ Risk Escalation             | Flags high-risk cases for human intervention                              |
| 📄 Session Summary             | Summarizes emotional cues and possible diagnoses for therapists            |
| 🔒 Safety Net                  | Secondary model prevents harmful outputs from GPT                         |

---

## 🔍 References

<a id="1">[1]</a> [Statistics Canada - Mental disorders](https://www150.statcan.gc.ca/n1/pub/12-581-x/2023001/sec8-eng.htm)  
<a id="2">[2]</a> [How Much Does Therapy Cost in Toronto?](https://thetherapycentre.ca/how-much-does-therapy-cost-in-toronto/)  
<a id="3">[3]</a> [Statistics Canada - Let’s Talk Mental Health](https://www.statcan.gc.ca/o1/en/plus/5461-lets-talk-mental-health)  
<a id="4">[4]</a> [Sentiment Analysis for Mental Health Dataset](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)  
<a id="5">[5]</a> [Suicidal Mental Health Dataset](https://www.kaggle.com/datasets/aradhakkandhari/suicidal-mental-health-dataset)  
<a id="6">[6]</a> [Mental Health Disorders Dataset](https://www.kaggle.com/datasets/kamaruladha/mental-disorders-identification-reddit-nlp)  
<a id="7">[7]</a> [Mental Health NLP Conversations Dataset](https://www.kaggle.com/datasets/thedevastator/nlp-mental-health-conversations)  
<a id="8">[8]</a> [Bidirectional LSTM for Sentiment Classification](https://aclanthology.org/P16-1034.pdf)  
