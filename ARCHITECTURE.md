# 🏗️ Architecture du Projet

Ce document présente l'architecture technique et la structure du projet de détection d'incidents.

---

## 1. Vue d'Ensemble

```
Model paramedic/
│
├── README.md                    # Documentation principale (orientée recruteur)
├── ARCHITECTURE.md              # Ce fichier
├── .gitignore                   # Fichiers à ignorer
│
├── docs/                        # Documentation détaillée
│   ├── METHODOLOGY.md           # Méthodologie complète
│   ├── THRESHOLD_OPTIMIZATION.md # Optimisation du seuil
│   └── RESULTS.md               # Résultats détaillés
│
└── notebooks/                   # Notebooks d'expérimentation
    ├── train_model.ipynb        # Entraînement du modèle CamemBERT
    └── test_seuil_perso3.ipynb  # Tests du seuil personnalisé
```

---

## 2. Pipeline de Traitement

### 2.1 Flux de Données

```
Données brutes (JSONL)
    ↓
Préprocessing
    ├── Encodage des labels (non_incident: 0, incident: 1)
    └── Tokenisation (CamemBERT, max_length=128)
    ↓
Dataset Hugging Face
    ├── Train (8,123 exemples)
    └── Test (2,031 exemples)
    ↓
Entraînement
    ├── Modèle: CamemBERT-base
    ├── Fine-tuning: 2 epochs
    └── Hyperparamètres optimisés
    ↓
Modèle entraîné
    ↓
Évaluation
    ├── Seuil standard (0.5)
    └── Seuil personnalisé (dynamique)
    ↓
Résultats et métriques
```

### 2.2 Composants Principaux

#### 2.2.1 Préprocessing

- **Format d'entrée** : JSONL (une ligne par exemple)
- **Champs utilisés** :
  - `text` : Message de communication
  - `label` : Label (non_incident / incident)
  - `trip_type` : Type de transport
  - `time_type` : Type de temps
  - `is_weekend` : Booléen
  - `is_bank_holidays` : Booléen
  - `dt_starting` : Heure de départ prévue
  - `first_message_dt` : Heure du premier message
  - `latest_message_dt` : Heure du dernier message

- **Tokenisation** :
  - Tokenizer : `CamembertTokenizer`
  - Max length : 128 tokens
  - Padding : `max_length`
  - Truncation : Activée

#### 2.2.2 Modèle

- **Architecture** : `CamembertForSequenceClassification`
- **Base** : `camembert-base` (Hugging Face)
- **Paramètres** : ~110M
- **Sortie** : 2 logits (classification binaire)

#### 2.2.3 Entraînement

- **Framework** : Hugging Face Transformers (Trainer)
- **Backend** : PyTorch
- **Hyperparamètres** : Voir [METHODOLOGY.md](docs/METHODOLOGY.md)

#### 2.2.4 Classification

- **Seuil standard** : 0.5 (par défaut)
- **Seuil personnalisé** : Dynamique selon le contexte
  - Base : 0.5
  - Réduction : -0.05 par critère de risque
  - Minimum : 0.3

---

## 3. Architecture du Modèle

### 3.1 CamemBERT

```
Input (Text)
    ↓
Tokenization (SentencePiece)
    ↓
Embeddings (Token + Position + Segment)
    ↓
Transformer Encoder (12 layers)
    ├── Multi-Head Attention (12 heads)
    ├── Feed Forward Network
    └── Layer Normalization
    ↓
[CLS] Token Representation (768 dim)
    ↓
Classification Head
    ├── Dense Layer (768 → 768)
    ├── Activation (ReLU)
    └── Output Layer (768 → 2)
    ↓
Logits (2 classes)
    ↓
Softmax
    ↓
Probabilités [P(non_incident), P(incident)]
```

### 3.2 Classification avec Seuil Personnalisé

```
Probabilités du modèle
    ↓
P(incident) = probas[:, 1]
    ↓
Calcul du seuil personnalisé
    ├── Seuil de base: 0.5
    ├── Réduction par critère de risque: -0.05
    └── Seuil minimum: 0.3
    ↓
Comparaison: P(incident) > seuil_personnalise ?
    ↓
Prédiction finale
```

---

## 4. Technologies et Dépendances

### 4.1 Bibliothèques Principales

- **transformers** (Hugging Face) : Modèles pré-entraînés et fine-tuning
- **datasets** (Hugging Face) : Gestion des datasets
- **torch** (PyTorch) : Backend de calcul
- **scikit-learn** : Métriques et évaluation
- **numpy** : Calculs numériques
- **matplotlib/seaborn** : Visualisations

### 4.2 Versions

*À préciser selon l'environnement utilisé*

- Python : 3.8+
- transformers : 4.x+
- torch : 1.x+
- scikit-learn : 1.x+

---

## 5. Structure des Données

### 5.1 Format d'Entrée (JSONL)

```json
{
  "text": "paramedic: Bonjour, un transporteur propose une PEC à 14h30...",
  "label": "incident",
  "trip_type": "Retour à domicile",
  "time_type": "Rendez-vous",
  "is_weekend": false,
  "is_bank_holidays": false,
  "dt_starting": "2025-07-01 07:45:00",
  "first_message_dt": "2025-07-01 07:42:26",
  "latest_message_dt": "2025-07-01 07:54:37",
  "ambulance_company": "Ambulances Clichy"
}
```

### 5.2 Format Après Tokenisation

```python
{
  "input_ids": [5, 1234, 5678, ...],  # Tokens encodés
  "attention_mask": [1, 1, 1, ...],    # Masque d'attention
  "label": 1                           # Label encodé (0 ou 1)
}
```

### 5.3 Format de Sortie

```python
{
  "predictions": [[logit_0, logit_1], ...],  # Logits bruts
  "label_ids": [0, 1, 0, ...],               # Labels réels
  "probabilities": [[0.8, 0.2], ...]         # Probabilités (softmax)
}
```

---

## 6. Flux d'Exécution

### 6.1 Phase d'Entraînement

1. **Chargement des données** : Lecture des fichiers JSONL
2. **Préprocessing** : Encodage des labels et tokenisation
3. **Initialisation du modèle** : Chargement de `camembert-base`
4. **Configuration de l'entraînement** : Hyperparamètres
5. **Entraînement** : Fine-tuning sur 2 epochs
6. **Sauvegarde** : Modèle et tokenizer sauvegardés

### 6.2 Phase d'Évaluation

1. **Chargement du modèle** : Modèle entraîné
2. **Préprocessing du test** : Tokenisation des données de test
3. **Prédiction** : Génération des probabilités
4. **Classification avec seuil standard** : Seuil fixe à 0.5
5. **Classification avec seuil personnalisé** : Seuil dynamique
6. **Évaluation** : Calcul des métriques (precision, recall, F1, accuracy)
7. **Visualisation** : Matrices de confusion

---

## 7. Points d'Extension

### 7.1 Améliorations Possibles

1. **Pipeline de production** :
   - API REST pour la prédiction en temps réel
   - Intégration dans un système de monitoring
   - Alertes automatiques

2. **Optimisation** :
   - Quantification du modèle (réduction de taille)
   - Optimisation pour l'inférence (ONNX, TensorRT)
   - Mise en cache des prédictions

3. **Monitoring** :
   - Tracking des performances en production
   - Détection de dérive (data drift)
   - A/B testing des seuils

4. **Amélioration du modèle** :
   - Fine-tuning continu (online learning)
   - Ensemble de modèles
   - Modèles spécialisés par type d'incident

---

## 8. Sécurité et Confidentialité

### 8.1 Données Sensibles

- ⚠️ **Aucune donnée confidentielle** : Les exemples présentés sont fictifs
- ⚠️ **Anonymisation** : Aucun nom réel d'entreprise ou de client
- ⚠️ **Conformité** : Respect des réglementations (RGPD, etc.)

### 8.2 Bonnes Pratiques

- **Versioning** : Git pour le code
- **Documentation** : Documentation complète du projet
- **Tests** : Validation sur données de test séparées
- **Reproductibilité** : Seeds fixes pour la reproductibilité

---

## 9. Déploiement

### 9.1 Environnement de Développement

- **Notebooks Jupyter** : Expérimentation et prototypage
- **Google Colab** : Entraînement sur GPU (si utilisé)

### 9.2 Production (Perspectives)

- **API REST** : Flask/FastAPI pour servir le modèle
- **Containerisation** : Docker pour l'isolation
- **Orchestration** : Kubernetes pour la scalabilité
- **Monitoring** : Logs et métriques de performance

---

## 10. Conclusion

Cette architecture présente un pipeline NLP complet et modulaire pour la détection d'incidents, avec une innovation majeure : **l'adaptation dynamique du seuil de classification au contexte métier**. La structure est conçue pour être extensible et maintenable, permettant des améliorations futures.

---

*Document basé sur l'analyse des notebooks et la méthodologie du projet*
