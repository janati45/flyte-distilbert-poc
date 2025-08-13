from __future__ import annotations
from typing import List, Tuple
import json
import tempfile
from pathlib import Path

import numpy as np
import datasets as hfds
from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import evaluate
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from flytekit import task, workflow, Resources
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile


# ========= Utils =========

def _detect_text_column(columns: List[str]) -> str:
    """Heuristique simple pour trouver la colonne texte."""
    for c in ["text", "tweet", "content", "sentence", "review", "message"]:
        if c in columns:
            return c
    # fallback: premier champ string
    return columns[0]


acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = acc_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
    return {"accuracy": acc, "f1_macro": f1}


def save_confusion_png(y_true, y_pred, labels: List[str], out_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    fig = plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45, ha="right")
    plt.yticks(ticks=range(len(labels)), labels=labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ========= Tasks =========

@task(cache=True, cache_version="v1")
def download_dataset(
    dataset_name: str = "tweet_eval",
    subset: str = "customer_support",
    val_ratio: float = 0.1,
    seed: int = 42,
) -> FlyteDirectory:
    """
    Télécharge le dataset HF, garantit train/val/test et sauvegarde sur disque.
    I/O sur disque => compatible S3/MinIO via Flyte.
    """
    ds = hfds.load_dataset(dataset_name, subset) if subset else hfds.load_dataset(dataset_name)
    # S'assure d'avoir validation
    if "validation" not in ds:
        split = ds["train"].train_test_split(test_size=val_ratio, seed=seed,
                                             stratify_by_column="label" if "label" in ds["train"].features else None)
        ds = DatasetDict({
            "train": split["train"],
            "validation": split["test"],
            "test": ds["test"] if "test" in ds else split["test"]
        })

    out_dir = Path(tempfile.mkdtemp(prefix="raw_ds_"))
    ds.save_to_disk(str(out_dir))
    return FlyteDirectory(path=str(out_dir))


@task(cache=True, cache_version="v1")
def tokenize_dataset(
    raw_ds_dir: FlyteDirectory,
    model_name: str = "distilbert-base-uncased",
    max_length: int = 128,
    text_column_hint: str = "",
) -> Tuple[FlyteDirectory, List[str], str]:
    """
    Tokenize en batch, retire les colonnes non utiles, format torch.
    Retourne: dir tokenisé, noms de labels, nom de la colonne texte retenue.
    """
    ds = hfds.load_from_disk(raw_ds_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    sample_split = list(ds.keys())[0]
    columns = ds[sample_split].column_names
    text_col = text_column_hint or _detect_text_column(columns)

    if "label" not in columns:
        raise ValueError("Le dataset doit contenir une colonne 'label'.")

    label_names = ds[sample_split].features["label"].names

    def tok(batch):
        return tokenizer(
            batch[text_col],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    remove_cols = [c for c in columns if c not in {text_col, "label"}]
    tok_ds = ds.map(tok, batched=True, remove_columns=remove_cols)
    tok_ds = tok_ds.with_format("torch")

    out_dir = Path(tempfile.mkdtemp(prefix="tok_ds_"))
    tok_ds.save_to_disk(str(out_dir))
    return FlyteDirectory(path=str(out_dir)), label_names, text_col


@task(
    cache=False,
    # Kubernetes: demandes & limites (adapter selon le cluster)
    requests=Resources(cpu="4", mem="16Gi"),
    limits=Resources(cpu="8", mem="24Gi"),
)
def train_and_eval(
    tok_ds_dir: FlyteDirectory,
    model_name: str,
    label_names: List[str],
    num_train_epochs: int = 2,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 32,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    evaluation_strategy: str = "epoch",
    save_strategy: str = "epoch",
    logging_steps: int = 50,
    seed: int = 42,
    report_to: str = "none",  # "wandb" ou "mlflow"
) -> Tuple[FlyteDirectory, FlyteFile, FlyteFile]:
    """
    Entraîne + évalue le modèle. Sauvegarde: modèle+tokenizer, metrics.json, confusion_matrix.png.
    Compatible S3/MinIO via FlyteDirectory/FlyteFile.
    """
    ds = hfds.load_from_disk(tok_ds_dir)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label_names)
    )

    # Dossiers de sortie (temp => Flyte uploadera ensuite vers le blobstore)
    base = Path(tempfile.mkdtemp(prefix="trainer_out_"))
    out_dir = base / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        evaluation_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        logging_steps=logging_steps,
        seed=seed,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to=[report_to] if report_to != "none" else [],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    val_metrics = trainer.evaluate()

    # Test
    preds = trainer.predict(ds["test"])
    test_logits = preds.predictions
    test_labels = preds.label_ids
    test_preds = np.argmax(test_logits, axis=-1)

    test_acc = acc_metric.compute(predictions=test_preds, references=test_labels)["accuracy"]
    test_f1 = f1_metric.compute(predictions=test_preds, references=test_labels, average="macro")["f1"]

    # Sauvegarde du modèle + tokenizer (serving-ready)
    model_dir = base / "model"
    model.save_pretrained(model_dir)
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tok.save_pretrained(model_dir)

    # Metrics
    metrics_path = base / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "val": val_metrics,
                "test": {"accuracy": test_acc, "f1_macro": test_f1},
                "labels": label_names,
            },
            f,
            indent=2,
        )

    # Confusion matrix
    cm_path = base / "confusion_matrix.png"
    save_confusion_png(test_labels, test_preds, label_names, cm_path)

    return (
        FlyteDirectory(path=str(model_dir)),
        FlyteFile(path=str(metrics_path)),
        FlyteFile(path=str(cm_path)),
    )


# ========= Workflow =========

@workflow
def classification_pipeline(
    dataset_name: str = "tweet_eval",
    subset: str = "customer_support",
    model_name: str = "distilbert-base-uncased",
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 32,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    max_length: int = 128,
    seed: int = 42,
    evaluation_strategy: str = "epoch",
    save_strategy: str = "epoch",
    logging_steps: int = 50,
    text_column_hint: str = "",
    report_to: str = "none",  # "wandb" ou "mlflow"
) -> Tuple[FlyteDirectory, FlyteFile, FlyteFile]:
    """
    Orchestration bout-en-bout.
    Retourne: répertoire du modèle HF, metrics.json, confusion_matrix.png
    """
    raw = download_dataset(dataset_name=dataset_name, subset=subset, seed=seed)
    tok_dir, label_names, _ = tokenize_dataset(
        raw_ds_dir=raw,
        model_name=model_name,
        max_length=max_length,
        text_column_hint=text_column_hint,
    )
    model_dir, metrics_json, cm_png = train_and_eval(
        tok_ds_dir=tok_dir,
        model_name=model_name,
        label_names=label_names,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        evaluation_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        logging_steps=logging_steps,
        seed=seed,
        report_to=report_to,
    )
    return model_dir, metrics_json, cm_png
