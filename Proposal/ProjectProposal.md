# The Mental Health Crisis is REAL

## What Is The Problem?  

Mental health care in Canada is facing a crisis marked by long wait times, unaffordable costs, and a persistent stigma that leaves millions without the support they need. In 2022, over 5 million Canadians aged 15 and older experienced significant symptoms of mental illness. Generalized anxiety disorder in this group tripled from 3.8% in 2012 to 11.9% in 2022, while major depressive episodes doubled from 9.0% to 18.4% [[1]](#1). Despite the rapidly increasing demand, many face wait times exceeding six weeks for mental health services, delaying critical care that could prevent these worsening conditions.  

In Toronto, the high cost of therapy further exacerbates the problem. Sessions typically range between $100 and $300 per hour, making care inaccessible for many &mdash; especially international students who often lack health insurance [[2]](#2). But the stigma surrounding mental health is the most pervasive barrier. Over 40% of Canadians fear judgment or negative repercussions for accessing mental health resources, deterring them from getting them the support they need [[3]](#3). These challenges create a system that fails to meet the mental health demands of its population, leading to increased rates of suicide.

**Addressing these issues requires an innovative solution that is accessible, affordable, and stigma-free, ensuring *no one* is left to navigate their struggles alone.**

---

## Why Do We Care?  

We weren’t there when it happened at the Bahen Centre [[4]](#4), but we’ve felt it. As university students ourselves, we know how mentally demanding our studies and relationships can be. The hollow ache of a bad day that turns into a bad month. The sleepless nights filled with unspoken worries, the quiet hope that maybe tomorrow will be easier. We’ve seen people we care deeply about retreat into solitude, suppressed by stigma and fear. We’ve seen the signs: the friend who stops showing up to class, the classmate whose laughter fades into whispers and then silence.

**We cannot let this continue. The stigma, the lengthy wait times, the system failing those who need it most &mdash; they’ve taken too much.** **It’s time to create something better.** With AI, we see an opportunity to break the cycle: to offer support without judgment, help without cost, and care without hesitation. 

---

## Why Machine Learning Can Help Solve This Problem  

Machine Learning offers a transformative solution to the challenges of mental health care. Unlike traditional systems constrained by waitlists and operating budgets, AI models/digital mental health platforms can provide essentially unlimited around-the-clock support at little to no cost. eliminating long wait times and financial burden [[5]](#5). The stigma associated with mental health care also diminishes when interacting with an AI system, as individuals may feel less judged by a machine than they might by a human.  

Our proposed solution would integrate two state-of-the-art NLP models: **FNet** for sentiment analysis and **Genrative Pre-trained Transformers** for customized response generation**. FNet is a Transformer-encoder architecture that utilizes Fourier transforms to efficiently process natural language inputs, identifying emotional cues with speed and accuracy &mdash; up to 80% faster than BERT and other transformer counterparts. [[6]](#6). This ensures that the system understands the user’s emotional state in real time. Building on FNet’s insights, our GPT would generate empathetic, context-aware responses specific to the user’s requirements. Together, these models form a cohesive system capable of validating emotions, offering coping mechanisms, and escalating critical cases to human professionals when necessary [[7]](#7). In light of the unpredicatbility of ML models, we felt it prudent to train a lightweight second classifier to flag malicious responses of the GPT Model. This should reduce the amount of poorly generated results while having a minimal impact on overall runtime.

**By leveraging machine learning, we aim to create a scalable and stigma-free mental health support system that meets individuals where they are, ensuring that no one is alone in their mental health struggle.**

---

## What Data To Use?  

Our model relies on carefully curated datasets designed to capture the complexities of human emotions and conversational context. These datasets enable the model to accurately identify emotional states and generate tailored responses:  

1. **Sentiment Analysis for Mental Health**: A dataset comprising **51074 text samples** with **two primary features**: the text data and corresponding mental health issue. The dataset categorizes sentiments related to mental health into distinct emotional labels, making it an essential resource for understanding sentiment trends, identifying emotional states, and contextualizing mental health-related discussions. It serves as a foundation for developing machine learning models that address mental health sentiment analysis and contributes to advancements in healthcare-focused NLP applications.[[8]](#8).  

2. **Suicidal Mental Health dataset**: The dataset includes 20,000 text samples, each classified under one of two labels: Depression or Suicide Watch. These labels provide a critical foundation for analyzing mental health concerns expressed in text.These capabilities can be instrumental in creating systems designed to prioritize cases for intervention, including escalating severe situations to emergency services.
This dataset contributes to the broader goal of enhancing mental health support systems by enabling the early detection of individuals in crisis and providing a pathway to timely assistance.[[9]](#9).

3. **Mental Disorders dataset**: The Mental Disorders dataset on Kaggle comprises 543060 text samples from reddit, each labeled with one of five mental health disorders: Bipolar Disorder, Depression, Borderline Personality Disorder, Anxiety and Schizophrenia. This dataset is structured in CSV format, with each row containing a text sample and its corresponding emotion label, making it suitable for training and evaluating machine learning models in natural language processing tasks such as sentiment analysis[[10]](#10).

4. **Mental Health NLP** : The Mental Health NLP dataset comprises approximately 2,500 entries, each featuring a conversation between a user and a psychologist, along with the corresponding response. This dataset is instrumental in training large language models to understand and generate contextually appropriate replies in mental health contexts. To mitigate the unpredictability of machine learning models, we propose training a secondary classifier to flag potentially harmful outputs from the primary GPT model, thereby enhancing response quality with minimal impact on runtime.[[11]](#11)

---

## What Do We Want from Let’s SOLVE it?  

We are inspired by Let’s SOLVE it’s mission to empower students to tackle real-world challenges using AI and machine learning. As undergraduate students who have experienced the mental health crisis firsthand, we believe this program offers the perfect opportunity to turn our vision into reality: a scalable, empathetic AI-driven mental health support system that breaks down barriers of access, cost, and stigma.  

1. **Mentorship**: Access to mentorship from experts will be invaluable in refining our approach to advanced machine learning techniques like sentiment analysis and natural language processing. Past participants, like Guy Morgenshtern, have highlighted how their mentor provided critical guidance and treated them like professionals, boosting their confidence and motivation [[10]](#10).  

2. **Prototype Development**: The program’s emphasis on turning ideas into viable prototypes aligns perfectly with our goal to build a functional system capable of recognizing emotions, generating empathetic responses, and escalating critical cases in real time.  

3. **Practical Experience**: Collaborating closely with industry professionals while still in school is a rare and transformative opportunity. Past participants, like India Tory, have emphasized the value of this experience, equipping them to tackle challenges beyond academia [[10]](#10).  

Let’s SOLVE it offers the platform we need to create something better and make a lasting difference. Together, let’s solve this.  

---

## References  
<a id="1">[1]</a>
[Statistics Canada - Mental disorders](https://www150.statcan.gc.ca/n1/pub/12-581-x/2023001/sec8-eng.htm)  

<a id="2">[2]</a>
[How Much Does Therapy Cost in Toronto?](https://thetherapycentre.ca/how-much-does-therapy-cost-in-toronto/)  

<a id="3">[3]</a>
[Statistics Canada - Let’s talk mental health](https://www.statcan.gc.ca/o1/en/plus/5461-lets-talk-mental-health) 

<a id="4">[4]</a>
[The Varsity - U of T Mental Health Crisis](https://thevarsity.ca/2019/09/29/enough-is-enough-this-is-an-emergency-u-of-t-must-immediately-address-its-mental-health-crisis/)  

<a id="5">[5]</a>
[AI in Mental Health](https://www.narcity.com/)  

<a id="6">[6]</a>
[FNet Model](https://arxiv.org/abs/2105.03824)  

<a id="7">[7]</a>
[GPT Model for Emotion Understanding](https://arxiv.org/pdf/2410.01306)  

<a id="8">[8]</a>
[Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)  

<a id="9">[9]</a>
[Suicidal Mental Health Dataset](https://www.kaggle.com/datasets/aradhakkandhari/suicidal-mental-health-dataset)  

<a id="10">[10]</a>
[Mental Health Disorder Dataset](https://www.kaggle.com/datasets/kamaruladha/mental-disorders-identification-reddit-nlp)

<a id="11">[11]</a>
[Mental Health NLP Dataset](https://www.kaggle.com/datasets/thedevastator/nlp-mental-health-conversations)

<a id="12">[12]</a>
[Let’s SOLVE it Success Stories](https://rbcborealis.com/lets-solve-it/)  
