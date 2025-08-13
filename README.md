# POC – Orchestration Flyte pour le fine-tuning de DistilBERT

## 1. Objectif du projet
Mettre en place un pipeline orchestré avec **Flyte** permettant de fine-tuner un modèle de langage pré-entraîné (par défaut `distilbert-base-uncased`) pour la classification de tickets de support client.  
L’objectif est de catégoriser les tickets dans différentes classes comme :  
- `complaint`  
- `request`  
- `comment`  

Le pipeline est **généralisable** à d’autres modèles Hugging Face (RoBERTa, CamemBERT, etc.) et à d’autres jeux de données texte/label.

## 2. Dataset utilisé

### 2.1 Nom et source
- **Nom** : `Customer Support on Twitter`
- **Source** : Collection TweetEval sur Hugging Face

### 2.2 Caractéristiques
- **Nombre de classes** : `complaint`, `request`, `comment`
- **Type de données** : tweets courts simulant des tickets clients
- **Avantages** :
  - Format proche des cas réels en entreprise
  - Compatible avec BERT-like models
  - Adapté aux tâches de classification multi-classes

## 3. Structure du pipeline Flyte

### 3.1 Étapes principales
1. **Téléchargement et préparation des données**
   - Chargement du dataset Hugging Face
   - Création d’un split `validation` si absent
   - Sauvegarde sur disque (`FlyteDirectory`)
2. **Tokenisation**
   - Détection de la colonne texte
   - Application de `AutoTokenizer` du modèle choisi
   - Padding et truncation (`max_length`)
   - Sauvegarde du dataset tokenisé
3. **Entraînement et évaluation**
   - Utilisation de `AutoModelForSequenceClassification`
   - Entraînement via `Trainer` (Transformers)
   - Évaluation : Accuracy, F1 macro
   - Génération d’une matrice de confusion
   - Sauvegarde du modèle et des artefacts

### 3.2 Diagramme du flux
download_dataset ──► tokenize_dataset ──► train_and_eval
| | ├─► model_dir/
| | ├─► metrics.json
| | └─► confusion_matrix.png



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
