# Vision Stock Pro - Application de Prédiction de Stock

## Description
Application Streamlit pour la prédiction intelligente de stock avec chatbot intégré.

## Structure du projet
```
Vision_Stock_Pro_Clean/
├── app.py                 # Application principale
├── requirements.txt       # Dépendances Python
├── packages.txt          # Dépendances système
├── README.md             # Ce fichier
├── data/                 # Données d'entraînement
│   ├── couche_softcqre_T4_clean.csv
│   ├── laitbroli_1kg_clean.csv
│   ├── may_arm_1kg_clean.csv
│   ├── may_arm_5kg_clean.csv
│   ├── papierhygsita_clean.csv
│   ├── parleG_clean.csv
│   └── sample_data_clean.csv
└── models/               # Modèles ML sauvegardés
    ├── gb_model.joblib
    ├── lgb_model.joblib
    ├── metadonnees.joblib
    ├── rf_model.joblib
    └── xgb_model.joblib
```

## Déploiement sur Streamlit Cloud
1. Uploadez ce dossier sur GitHub
2. Connectez-le à Streamlit Cloud
3. Configurez les variables d'environnement si nécessaire

## Variables d'environnement
- `GROQ_API_KEY` : Clé API Groq pour le chatbot (optionnelle)
