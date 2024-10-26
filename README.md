# Customer-Order Query Generation using Transformers
This repository provides a demonstration of training a sequence-to-sequence language model using the BART transformer to generate answers to natural language questions based on customer-order relationships in tabular data. It uses question-answer pairs as examples to train a model that can answer business queries in JSON format.


## Overview
This project leverages the BART model for sequence-to-sequence learning, training it on question-answer pairs that contain both natural language queries and tabular data specifications. The objective is to answer questions about customer data, providing structured JSON output describing customer information and orders.

### Key Components:
* Data: Contains structured tabular data information and their relationships.
* Model: A BART model fine-tuned on question-answer pairs to produce JSON responses.
* Training: Custom training with tokenization, batching, and parameter adjustments.
## Setup
Clone this repository:

```
git clone https://github.com/Kamaleshtue23/Transformer-Customer-Query(Attentions.ai).git
```
cd transformer-customer-query
Install dependencies:
```
pip install transformers[torch] pandas datasets
```
## Data Preparation
The dataset includes example question-answer pairs in JSON format:

* **Questions:** Structured natural language queries (e.g., “Give me the top 5 customers with the most number of orders.”)
* **Answers:** JSON-formatted responses including table information and relationships among tables.
### Sample data schema:
```
qa_pairs = [
    {
        "question": "Give me the top 5 customers with the most number of orders.",
        "answer": {
            "table_information": [
                {
                    "table_name": "customer_0",
                    "columns": [
                        {"name": "customer_id", "data_type": "STRING", "is_pii_column": "Y"},
                        {"name": "first_name", "data_type": "STRING", "is_pii_column": "Y"},
                        {"name": "last_name", "data_type": "STRING", "is_pii_column": "Y"}
                    ]
                },
                {
                    "table_name": "order",
                    "columns": [
                        {"name": "order_id", "data_type": "STRING"},
                        {"name": "customer_id", "data_type": "STRING", "is_pii_column": "Y"}
                    ]
                }
            ],
            "relationships": [
                {"FromTable": "customer_0", "FromColumn": ["customer_id"], "ToTable": "order", "ToColumn": ["customer_id"]}
            ]
        }
    }
]
```
## Model Training
The project uses a pre-trained BART model from the Hugging Face Transformers library. To fine-tune the model:

1. Tokenization: Prepare the inputs and target JSON responses for training.

2. Training Configuration:
* Define training arguments (batch size, learning rate, and number of epochs).
* Use a Trainer instance for model training.
### Code Example
```
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd
import json

# Load data and define the model
dataset = Dataset.from_pandas(pd.DataFrame(qa_pairs))
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large")

def preprocess_function(examples):
    inputs = examples['question']
    targets = [json.dumps(ans) for ans in examples['answer']]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=512, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets
)

# Train the model
trainer.train()
```
## Results
After training, the model can answer business-related queries in JSON format, specifying detailed table and relationship information based on the provided questions.

## Requirements
* `transformers`
* `torch`
* `pandas`
* `datasets`
Install these with:

```
pip install transformers[torch] pandas datasets
```
## Acknowledgments
This project uses the Hugging Face Transformers library and the BART model, which are instrumental for natural language processing and sequence-to-sequence tasks.
