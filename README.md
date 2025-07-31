# POC Flyte - Fine-tuning DistilBERT pour classification de tickets clients

## Objectif
Ce projet démontre comment orchestrer un pipeline de fine-tuning de `distilbert-base-uncased` pour classifier des tickets de support client en catégories telles que : "complaint", "request", "comment", à l’aide de Flyte.

## Structure du pipeline
- Téléchargement du dataset "Customer Support on Twitter" (`TweetEval`)
- Tokenisation via Hugging Face
- Séparation train / val / test
- Fine-tuning avec `Trainer`
- Sauvegarde du modèle

## Exécution locale avec Flyte Sandbox

### Prérequis
- Python ≥ 3.8
- Docker
- pip (`pip install flytectl`)

### Étapes
```bash
flytectl demo start
flytectl register files --project flytesnacks --domain development --archive .
```

Interface Flyte : http://localhost:30081/console

## À venir
- Intégration MinIO/S3
- Monitoring avec Weights & Biases
- Déploiement modèle