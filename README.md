# 🚑 Détection d'Incidents dans les Communications de Transport Médical

> Projet de classification NLP utilisant CamemBERT pour identifier automatiquement les incidents dans les échanges de communication entre transporteurs médicaux et opérateurs.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-orange.svg)](https://huggingface.co/)
[![CamemBERT](https://img.shields.io/badge/Model-CamemBERT-green.svg)](https://huggingface.co/camembert-base)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Portfolio-yellow.svg)](https://github.com)

**🎯 Résultat clé** : Réduction des faux négatifs de **124 à 1** (-99%) grâce à l'optimisation du seuil de classification.

## 📋 Résumé Exécutif

Ce projet présente un système de classification binaire utilisant le modèle de langue français **CamemBERT** pour détecter automatiquement les incidents dans les communications de transport médical. L'innovation principale réside dans l'implémentation d'un **seuil de classification personnalisé et dynamique** qui s'adapte aux contextes métier, permettant de réduire significativement les faux négatifs tout en maintenant une précision élevée.

### 🎯 Objectifs Métier

- **Réduction des faux négatifs** : Minimiser le risque de ne pas détecter un incident réel (critique dans le domaine médical)
- **Optimisation du seuil de classification** : Adaptation dynamique selon le contexte (type de transport, horaires, jours fériés, etc.)
- **Automatisation** : Détection en temps réel des incidents pour améliorer la réactivité opérationnelle

### 📊 Résultats Clés

| Métrique | Seuil Standard (0.5) | Seuil Optimal (0.90) | Amélioration |
|----------|----------------------|---------------------|--------------|
| **Accuracy Globale** | ≈ 90% | ≈ 90% | Stable |
| **F1-Score Global** | 0.91 | 0.91 | Stable |
| **F1-Score Incidents** | 0.73 | Amélioré | + |
| **Recall (Incident)** | Faible | **Beaucoup plus élevé** | **+++** |
| **Faux Négatifs** | Nombreux | **Réduits significativement** | **Réduction majeure** |

> **Note** : Le choix métier privilégie le recall élevé pour les incidents, acceptant une augmentation des faux positifs afin de garantir qu'aucun incident réel ne soit manqué.

---

## 🏗️ Architecture du Projet

```
Model paramedic/
│
├── README.md                 # Ce fichier
├── ARCHITECTURE.md           # Architecture technique
├── docs/                     # Documentation détaillée
│   ├── METHODOLOGY.md        # Méthodologie complète
│   ├── THRESHOLD_OPTIMIZATION.md  # Optimisation du seuil
│   └── RESULTS.md            # Résultats détaillés
│
├── notebooks/                # Notebooks d'expérimentation
│   ├── train_model.ipynb     # Entraînement du modèle CamemBERT
│   └── test_seuil_perso3.ipynb  # Tests du seuil personnalisé
│
├── assets/                   # Images et visualisations
│   ├── training_results.png
│   ├── confusion_matrix_standard.png
│   └── confusion_matrix_custom_threshold.png
│
└── .gitignore
```

---

## 🔬 Méthodologie

### 1. Modèle de Base : CamemBERT

- **Modèle** : `camembert-base` (Hugging Face)
- **Architecture** : Transformer BERT adapté au français
- **Tâche** : Classification binaire (incident / non_incident)
- **Fine-tuning** : 2 epochs avec learning rate 2e-5

#### Hyperparamètres d'Entraînement

```python
TrainingArguments(
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    warmup_steps=200,
    lr_scheduler_type="cosine",
    metric_for_best_model="accuracy"
)
```

#### Dataset

- **Train** : 8,123 exemples
- **Test** : 2,031 exemples
- **Format** : JSONL avec champs `text` et `label`
- **Labels** : `non_incident` (0) / `incident` (1)

### 2. Innovation : Optimisation du Seuil de Classification

Le seuil standard (0.5) produisait trop de faux négatifs. Après analyse des courbes précision-rappel-F1, un **seuil optimal de 0.90** a été identifié, permettant de maximiser le rappel (détection des incidents) tout en maintenant une précision acceptable.

#### Approche 1 : Seuil Fixe Optimal

- **Seuil standard** : 0.5 → Trop de faux négatifs
- **Seuil optimal** : 0.90 → Rappel beaucoup plus élevé, faux négatifs réduits significativement

#### Approche 2 : Personnalisation Dynamique (Expérimentée)

Une personnalisation dynamique du seuil a également été expérimentée, en fonction de paramètres de risque identifiés :
- Type de trajet
- Contexte week-end/jour férié
- Timing des messages

Cette approche a permis de réduire fortement les faux négatifs, tout en gardant les faux positifs sous contrôle.

---

## 📈 Résultats Détaillés

### Performance avec Seuil Standard (0.5)

- **Accuracy** : ≈ 90%
- **F1-Score Global** : 0.91
- **F1-Score Incidents** : 0.73
- **Problème** : **124 faux négatifs** (incidents réels non détectés) ⚠️

![Matrice de confusion - Seuil standard](assets/confusion_matrix_standard.png)

### Performance avec Seuil Personnalisé

- **Accuracy** : ≈ 90% (maintenue)
- **Rappel (Recall)** : 0.95 (vs 0.67 avec seuil standard)
- **Faux négatifs** : **1 seul** (vs 124 avec seuil standard) ✅
- **Compromis** : Rappel élevé (peu d'incidents oubliés) avec précision plus faible (plus de faux positifs)

![Matrice de confusion - Seuil personnalisé](assets/confusion_matrix_custom_threshold.png)

### Analyse

- ✅ **Rappel incident** : Amélioration significative
- ✅ **Faux négatifs** : Réduction majeure
- ⚠️ **Précision incident** : Plus faible (trade-off accepté pour maximiser la détection)
- ✅ **Accuracy globale** : Maintenue à ≈ 90%

---

## 💡 Choix Techniques et Justifications

### Pourquoi CamemBERT ?

- **Spécialisé français** : Entraîné sur un large corpus français
- **Performance** : État de l'art pour les tâches NLP en français
- **Intégration** : Facilement intégrable via Hugging Face Transformers

### Pourquoi un Seuil Personnalisé ?

Dans le contexte médical, **un faux négatif (incident non détecté) est bien plus critique qu'un faux positif**. Le seuil personnalisé permet de :

1. **Réduire drastiquement les faux négatifs** : De ~124 à ~1
2. **S'adapter au contexte** : Prise en compte des facteurs de risque métier
3. **Maintenir l'accuracy globale** : Impact minimal sur la performance globale

### Trade-off Precision/Recall

Le choix métier privilégie le **recall élevé** pour les incidents :
- **Seuil 0.5** : Nombreux faux négatifs (incidents non détectés)
- **Seuil 0.90** : Rappel beaucoup plus élevé, faux négatifs réduits significativement

Cette approche garantit qu'aucun incident critique ne passe inaperçu, même si cela génère plus d'alertes à vérifier manuellement (faux positifs acceptables).

---

## 🛠️ Technologies Utilisées

- **Python** 3.8+
- **Transformers** (Hugging Face) : Modèles pré-entraînés
- **Datasets** (Hugging Face) : Gestion des données
- **scikit-learn** : Métriques et évaluation
- **PyTorch** : Backend de calcul
- **matplotlib/seaborn** : Visualisations

---

## 📚 Documentation Complète

Pour plus de détails, consultez :

- **[Méthodologie détaillée](docs/METHODOLOGY.md)** : Pipeline complet, préprocessing, entraînement
- **[Optimisation du seuil](docs/THRESHOLD_OPTIMIZATION.md)** : Algorithme, facteurs de risque, exemples
- **[Résultats](docs/RESULTS.md)** : Métriques détaillées, matrices de confusion, analyses

---

## 🎓 Compétences Développées

Ce projet démontre :

- ✅ **NLP avancé** : Fine-tuning de modèles transformer (CamemBERT)
- ✅ **Classification binaire** : Optimisation pour cas d'usage métier
- ✅ **Ingénierie des features** : Intégration de features métier (contexte temporel, type de transport)
- ✅ **Optimisation métier** : Trade-off precision/recall adapté au domaine
- ✅ **Évaluation** : Métriques adaptées au contexte (focus sur recall)
- ✅ **Python/ML** : Transformers, scikit-learn, PyTorch

---

## 📝 Notes Importantes

- ⚠️ **Aucune donnée confidentielle** : Les exemples présentés sont fictifs
- ⚠️ **Projet portfolio** : Ce dépôt est une vitrine technique, non exécutable
- ⚠️ **Source de vérité** : Les résultats et méthodologie sont basés sur le rapport de stage (source canonique)
- ⚠️ **Données confidentielles** : Aucune donnée réelle de l'entreprise n'est présente dans ce dépôt
- 📚 **Documentation complète** : Voir le dossier `docs/` pour les détails techniques

## 🚀 Installation (Pour référence uniquement)

Ce projet est présenté à des fins de démonstration. Pour reproduire l'environnement :

```bash
pip install -r requirements.txt
```

**Note** : Les notebooks nécessitent un accès GPU (Google Colab recommandé) pour l'entraînement.

---

## 👤 Auteur

**Mohamed Ben Amor**  
Stage Année 1 - Projet de Classification NLP

---

## 📄 Licence

Ce projet est présenté à des fins de démonstration et de portfolio.

---

## 🔗 Références

- [CamemBERT](https://huggingface.co/camembert-base) - Modèle de langue français
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) - Bibliothèque NLP
- [scikit-learn](https://scikit-learn.org/) - Machine Learning en Python

---

## 📌 Topics GitHub Recommandés

Pour améliorer la découvrabilité sur GitHub, ajoutez ces topics :
- `nlp`
- `camembert`
- `transformers`
- `classification`
- `french-nlp`
- `machine-learning`
- `deep-learning`
- `huggingface`
- `portfolio`
- `medical-ai`

---

*Dernière mise à jour : Janvier 2026*
