# 🧠 Next-Word Predictor using GPT-2

Welcome to my **final project** for the NLP course!  
This project is a simple and elegant implementation of a **Next-Word Prediction system** using the powerful GPT-2 model from Hugging Face Transformers.  
The model takes a user-provided prompt and predicts how the sentence is likely to continue ✨.

---

## 📋 Project Overview

> **Goal:** Build a text generation model that can predict the next word(s) given an input sentence or phrase.

This project focuses on:
- 🔎 Tokenizing and formatting a real-world dataset (`WikiText-2`)
- 🧠 Fine-tuning the GPT-2 language model
- 🛠️ Implementing a next-word prediction function
- 🗣️ Generating natural text based on a seed input

---

## 🛠️ Tech Stack

- **Python** 🐍
- **Hugging Face Transformers** 🤗
- **Datasets Library** 📦
- **PyTorch** 🔥
- **VS Code / Jupyter** 💻

---

## 📚 Dataset Used

The model is trained using the **WikiText-2** dataset:
- 📖 Clean and curated Wikipedia text
- ✂️ Limited to the first 1000 lines for fast, beginner-friendly training
- 🚀 Lightweight enough to run on a CPU

---

## 🧪 Model Training Steps

```python
# Step-by-step process:
1. Load Dataset         → `load_dataset("wikitext", "wikitext-2-raw-v1")`
2. Tokenize Text        → using `AutoTokenizer` from GPT-2
3. Format Input Blocks  → fixed length using a grouping function
4. Load GPT-2 Model     → `GPT2LMHeadModel.from_pretrained("gpt2")`
5. Train Model          → with `Trainer` API
6. Predict Next Words   → using `.generate()` function

🤖 Sample Prediction:
Input  ➤ "The future of artificial intelligence is"
Output ➤ "The future of artificial intelligence is not just promising but inevitable in shaping our world."
It continues your sentence in a realistic and fluent way 💬

🚀 How to Run:

✅ Install Dependencies
bash
Copy
Edit
pip install transformers datasets torch

✅ Run the Script
Copy the code from main.py or run the .ipynb notebook.

Use the predict_next_word() function to generate completions.

📈 Customizations
Feel free to:

Use larger GPT-2 models (gpt2-medium, gpt2-large)

Expand dataset or epochs for better results

Add a Streamlit/Gradio interface for real-time generation


🧠 What I Learned:
How transformer models tokenize and understand language

Practical usage of Hugging Face Trainer API

Preprocessing and batching real datasets

Building user-facing functions for intelligent output

🙋 Author Info
Name: [Your Name]
Institute: IIT Bombay
Course Track: NLP (Transformers Track)
Project: Final Submission — Week 4

⭐ Final Words
This project marks the completion of my NLP course journey — starting from scratch to building a working text generation model with real AI 🔥

If you're curious, experiment and build — you’ll learn more than you ever expect! 🚀

