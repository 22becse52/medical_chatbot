# Medical Diagnosis Chatbot using NLP

An intelligent **Medical Chatbot** powered by **DistilBERT (Transformers)** that diagnoses diseases based on user symptoms and provides **precautionary measures** and **diet recommendations**.  
This project combines **Natural Language Processing (NLP)** and **Deep Learning** to create a healthcare assistant that can analyze human health-related queries and respond intelligently.

```

##  Features
Understands natural language medical queries  
Predicts disease name from symptoms  
Provides precautions and diet recommendations  
rained on 1000+ labeled queries for 50+ diseases  
Achieved **94% validation accuracy**  
Built and trained on **Google Colab**  
Ready for **API deployment** via FastAPI or Flask  

---

## Tech Stack
- **Language:** Python  
- **Libraries:** Transformers, PyTorch, Scikit-learn, Pandas, Matplotlib  
- **Model:** DistilBERT (Pretrained Transformer)  
- **Dataset:** Custom medical dataset (CSV format)  
- **Environment:** Google Colab / Jupyter Notebook

```

## Project Structure

```
medical-chatbot/
│
├── medical_chatbot_dataset.csv # Custom dataset with 1000+ queries
├── train_chatbot.ipynb # Colab training notebook
├── chatbot_interface.py # Chat testing script
├── app.py # FastAPI/Flask API for deployment
├── chatbot_model/ # Saved fine-tuned model
└── README.md # Project documentation

```
```

##  Dataset Description

The dataset includes user queries describing symptoms, mapped to corresponding diseases, precautions, and diet plans.

| Column | Description |
|---------|--------------|
| **query** | User symptom or question |
| **diagnosis** | Predicted disease label |
| **precautions** | Precautionary measures |
| **diet_plan** | Suggested food and diet |

**Example Entry:**

| Query | Diagnosis | Precautions | Diet Plan |
|--------|------------|--------------|------------|
| I have cough and sore throat | Common Cold | Rest, drink warm fluids | Soup, fruits, hydration |
```
##  Model Overview

**Model Used:** [DistilBERT Base Uncased](https://huggingface.co/distilbert-base-uncased)

DistilBERT is a smaller, faster version of BERT that retains 97% of its language understanding performance while reducing model size and computation time.

**Model Training Arguments:**
```python
evaluation_strategy="epoch"
learning_rate=2e-5
batch_size=8
num_train_epochs=3
weight_decay=0.01
```
```bash
!pip install transformers datasets torch scikit-learn pandas matplotlib
```

Model Performance
Metric	Score
Accuracy	94.2%
F1 Score	0.93
Precision	0.94
Recall	0.92

