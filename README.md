
## 4. Détails du fine-tuning

### 4.1 Modèle
- `AutoModelForSequenceClassification`  
- Nombre de labels déterminé automatiquement via `features["label"].names`

### 4.2 Tokenisation
- `AutoTokenizer` (version `use_fast=True`)
- Paramètres clés :
  - `max_length` : longueur maximale de séquence
  - `padding` : `max_length`
  - `truncation` : `True`

### 4.3 Entraînement
- API `Trainer` de Hugging Face Transformers
- Paramètres configurables :
  - `num_train_epochs`
  - `per_device_train_batch_size`
  - `per_device_eval_batch_size`
  - `learning_rate`
  - `weight_decay`
  - `evaluation_strategy`, `save_strategy`
  - `seed`

### 4.4 Évaluation
- Métriques :
  - Accuracy
  - F1 macro
- Matrice de confusion :
  - Format PNG
  - Classes ordonnées selon `label_names`

## 5. Technologies utilisées

### 5.1 Orchestration
- [Flyte](https://flyte.org/) pour la gestion du pipeline et des tâches

### 5.2 NLP
- [Hugging Face Transformers](https://huggingface.co/transformers/)  
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/)

### 5.3 Backend
- PyTorch

### 5.4 Environnement
- Containerisation avec Docker
- Stockage d’artefacts avec MinIO ou AWS S3

## 6. Installation et exécution locale

### 6.1 Prérequis
- Python ≥ 3.10
- Docker (optionnel)
- Flyte CLI :
```bash
pip install flytectl pyflyte
