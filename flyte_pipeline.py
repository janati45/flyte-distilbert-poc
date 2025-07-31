from flytekit import task, workflow
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

@task
def download_data():
    dataset = load_dataset("tweet_eval", "customer_support")
    return dataset

@task
def tokenize_data(dataset: dict, model_name: str = "distilbert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def preprocess(example):
        return tokenizer(example["text"], truncation=True, padding="max_length")
    tokenized_dataset = dataset.map(preprocess, batched=True)
    return tokenized_dataset

@task
def split_data(tokenized_dataset: dict):
    train = tokenized_dataset["train"]
    val = tokenized_dataset["validation"]
    test = tokenized_dataset["test"]
    return train, val, test

@task
def train_model(train, val, model_name: str = "distilbert-base-uncased"):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {
            "accuracy": accuracy_score(p.label_ids, preds),
            "f1": f1_score(p.label_ids, preds, average="weighted")
        }

    args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=val,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model("./model")
    return "model saved"

@workflow
def classification_pipeline():
    dataset = download_data()
    tokenized = tokenize_data(dataset=dataset)
    train, val, test = split_data(tokenized_dataset=tokenized)
    train_model(train=train, val=val)