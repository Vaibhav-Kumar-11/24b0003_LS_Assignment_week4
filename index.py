# Load dataset (i am using a built-in simple dataset WikiText2)
from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_texts = dataset["train"]["text"][:1000]  


# Loading tokenizer and tokenizing the dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

def tokenize_function(example):
    return tokenizer(example["text"], return_special_tokens_mask=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])


# Grouping tokens for training
block_size = 128 # This basically helps create input sequences of uniform length like the random_state = 42 type of case.

def group_texts(examples):
    concatenated = sum(examples["input_ids"], [])
    total_length = (len(concatenated) // block_size) * block_size
    result = {
        "input_ids": [concatenated[i:i + block_size] for i in range(0, total_length, block_size)],
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_dataset = tokenized_dataset.map(group_texts, batched=True)


# Loading pre-trained GPT2 model
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained("gpt2")


# Training preparation - using Trainer API (
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./gpt2-nextword",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    logging_steps=50,
    save_steps=200,
    save_total_limit=2,
    report_to="none",  # disables wandb or hub
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"]
)


# Training the model
trainer.train()


# Writing the function to predict next word(s)
import torch

def predict_next_word(prompt_text, max_new_tokens=10):
    inputs = tokenizer(prompt_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True)
    return tokenizer.decode(outputs[0])

# Example:
print(predict_next_word("The future of artificial intelligence is"))
