---
title: Credit Risk Dashboard
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
port: 7860
---

# Credit Risk Dashboard

Dashboard de scoring crédit avec stockage externe des assets sur HuggingFace Hub.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    HuggingFace Spaces                           │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Docker Container (~50 MB)                     │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐    │  │
│  │  │  Streamlit  │  │   FastAPI   │  │  InferenceService│   │  │
│  │  │  (7860)     │──│   (8000)    │──│  + HFHubManager  │   │  │
│  │  └─────────────┘  └──────┬──────┘  └─────────────────┘    │  │
│  └──────────────────────────┼────────────────────────────────┘  │
│                             │ Download at startup                │
│                             ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │               HuggingFace Hub Repos                       │   │
│  │  ┌────────────────────┐  ┌─────────────────────────────┐  │   │
│  │  │ Model Repository   │  │ Dataset Repository          │  │   │
│  │  │ - MLflow model     │  │ - application_test.csv      │  │   │
│  │  │   (~480 KB)        │  │ - shap_explanation.joblib   │  │   │
│  │  │                    │  │ - shap_beeswarm.png         │  │   │
│  │  │                    │  │ - plot JSON files           │  │   │
│  │  └────────────────────┘  └─────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Différences avec le dépôt original

| Aspect | Original | Refactored |
|--------|----------|------------|
| Stockage assets | Git LFS dans le repo | HuggingFace Hub |
| Taille image Docker | ~150 MB | ~50 MB |
| Mise à jour modèle | Rebuild image | `huggingface-cli upload` |
| Premier démarrage | Instant | +30s (téléchargement) |

## Configuration requise

### Variables d'environnement (HF Space Settings)

| Variable | Description | Exemple |
|----------|-------------|---------|
| `HF_MODEL_REPO_ID` | Dépôt HF Hub du modèle | `username/credit-risk-dashboard-model` |
| `HF_DATA_REPO_ID` | Dépôt HF Hub des données | `username/credit-risk-dashboard-data` |

### Secrets GitHub Actions

| Secret | Description |
|--------|-------------|
| `HF_TOKEN` | Token d'accès HuggingFace (write) |
| `HF_USERNAME` | Nom d'utilisateur HuggingFace |

## Structure des dépôts HF Hub

### Dépôt modèle (`credit-risk-dashboard-model`)

```
gradient_boosting/
├── MLmodel
├── model.pkl
├── conda.yaml
├── python_env.yaml
├── requirements.txt
└── code/
    └── p7_utils/
        ├── __init__.py
        ├── config.py
        ├── logs.py
        └── metrics.py
```

### Dépôt dataset (`credit-risk-dashboard-data`)

```
├── application_test.csv
├── shap/
│   ├── shap_explanation.joblib
│   └── shap_beeswarm.png
└── plots/
    ├── DAYS_EMPLOYED_hist_data.json
    ├── EXT_SOURCE_2_hist_data.json
    ├── EXT_SOURCE_3_hist_data.json
    └── OWN_CAR_AGE_hist_data.json
```

## Déploiement

1. **Créer les 2 dépôts HF Hub** et uploader les assets manuellement via l'interface web

2. **Configurer le HF Space** `credit_scoring_dashboard_refactored` :
   - Settings → Variables : ajouter `HF_MODEL_REPO_ID` et `HF_DATA_REPO_ID`
   - (Optionnel) Settings → Persistent Storage : activer pour cache

3. **Configurer GitHub** :
   - Settings → Secrets : ajouter `HF_TOKEN` et `HF_USERNAME`

4. **Push sur main** → Déploiement automatique

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /customers` | Liste des IDs clients |
| `GET /customer/{id}/dashboard` | Données complètes (composite) |
| `GET /customer/{id}/score` | Score pour gauge |
| `GET /customer/{id}/features` | 4 features principales |
| `GET /customer/{id}/shap` | Valeurs SHAP locales |
| `GET /features/bivariate_data` | Données scatter plot |
| `GET /shap/global` | Image beeswarm SHAP |
| `GET /features/{feature}/distribution` | Histogramme |

## Licence

MIT
