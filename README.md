OC Flyte – Fine-tuning BERT/DistilBERT pour classification de tickets clients
Objectif
Ce projet montre comment mettre en place, avec Flyte, un pipeline complet de fine-tuning d’un modèle de langage pré-entraîné (par défaut distilbert-base-uncased) pour la classification automatique de tickets de support client.
Les tickets sont catégorisés en classes telles que complaint, request ou comment.
Le pipeline est entièrement généralisable à d’autres modèles et datasets texte/label disponibles sur Hugging Face.

Structure du pipeline
Le pipeline se décompose en trois étapes principales :

Téléchargement et préparation des données

Chargement du dataset Hugging Face (ex. tweet_eval, subset customer_support)

Création d’un split validation si nécessaire

Sauvegarde au format disque (FlyteDirectory)

Tokenisation

Chargement d’un tokenizer (AutoTokenizer) associé au modèle choisi

Détection automatique de la colonne texte

Tokenisation avec padding/truncation (max_length)

Sauvegarde du dataset tokenisé

Entraînement et évaluation

Chargement du modèle (AutoModelForSequenceClassification)

Entraînement avec Trainer de Hugging Face Transformers

Évaluation (accuracy, F1 macro)

Génération et sauvegarde d’une matrice de confusion

Sauvegarde du modèle et du tokenizer

Diagramme simplifié
markdown
Copy
Edit
download_dataset ──► tokenize_dataset ──► train_and_eval
      |                    |                     ├─► model_dir/
      |                    |                     ├─► metrics.json
      |                    |                     └─► confusion_matrix.png
Contenu du projet
bash
Copy
Edit
.
├─ pipeline.py          # Code Flyte : tâches + workflow principal
├─ requirements.txt     # Dépendances Python
├─ Dockerfile           # Optionnel : build image pour exécution en cluster
└─ README.md            # Documentation
Prérequis
Python ≥ 3.10

Docker

pip (pip install flytectl pyflyte)

Installation locale
bash
Copy
Edit
pip install -r requirements.txt
Exécution locale (sans cluster)
bash
Copy
Edit
pyflyte run pipeline.py orchestrate_workflow \
  --dataset_name tweet_eval \
  --subset customer_support \
  --model_name distilbert-base-uncased \
  --num_train_epochs 1 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 32 \
  --learning_rate 5e-5 \
  --weight_decay 0.01 \
  --max_length 128 \
  --seed 42
Sorties :

model_dir/ : modèle et tokenizer au format Hugging Face

metrics.json : métriques sur validation et test

confusion_matrix.png : matrice de confusion sur le jeu de test

Exécution avec Docker
bash
Copy
Edit
docker build -t flyte-bert-poc .
docker run --rm -it flyte-bert-poc
Utilisation avec Flyte Sandbox / Cluster
1. Démarrer la sandbox
bash
Copy
Edit
flytectl demo start
2. Enregistrer le workflow
bash
Copy
Edit
pyflyte register . --project poc --domain development
3. Lancer une exécution
Exemple avec un fichier exec.yaml :

yaml
Copy
Edit
inputs:
  dataset_name: "tweet_eval"
  subset: "customer_support"
  model_name: "distilbert-base-uncased"
  num_train_epochs: 1
  per_device_train_batch_size: 16
  per_device_eval_batch_size: 32
  learning_rate: 5e-5
  weight_decay: 0.01
  max_length: 128
  seed: 42
Lancement :

bash
Copy
Edit
flytectl create execution --project poc --domain development \
  --exec-file exec.yaml --workflow orchestrate_workflow
Interface Flyte : http://localhost:30081/console

Paramètres principaux
Paramètre	Description
dataset_name	Nom du dataset Hugging Face
subset	Sous-jeu éventuel
model_name	Modèle Hugging Face (distilbert-base-uncased, roberta-base, camembert-base, etc.)
max_length	Longueur maximale des séquences tokenisées
num_train_epochs	Nombre d’époques d’entraînement
per_device_train_batch_size	Taille de batch pour l’entraînement
per_device_eval_batch_size	Taille de batch pour l’évaluation
learning_rate	Taux d’apprentissage
weight_decay	Décroissance de poids
text_column_hint	Nom de la colonne texte si non détectée automatiquement
report_to	"none", "wandb" ou "mlflow" pour le suivi des runs

Généralisation
Changer de modèle
bash
Copy
Edit
--model_name roberta-base
Pour un modèle francophone :

bash
Copy
Edit
--model_name camembert-base
Changer de dataset
Exemple avec Allociné :

bash
Copy
Edit
pyflyte run pipeline.py orchestrate_workflow \
  --dataset_name allocine \
  --model_name camembert-base \
  --max_length 256 \
  --num_train_epochs 2 \
  --text_column_hint review
Adapter à un autre cas
Le pipeline est compatible avec tout dataset contenant une colonne texte et une colonne label.
Le nombre de classes est détecté automatiquement via features["label"].names.

Extensions possibles
Hyperparameter sweep avec un workflow dynamique Flyte

Intégration MinIO / S3 pour le stockage d’artefacts

Monitoring avec Weights & Biases ou MLflow

Support GPU dans train_and_eval

Ajout d’une étape d’inférence (FastAPI)

Monitoring du drift et retrain automatique
