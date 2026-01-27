# 📖 Méthodologie Détaillée

Ce document présente la méthodologie complète du projet de détection d'incidents dans les communications de transport médical.

---

## 1. Contexte et Objectifs

### 1.1 Problématique

Dans le domaine du transport médical, les communications entre transporteurs et opérateurs peuvent contenir des informations critiques sur des incidents (retards, pannes, problèmes de transport, etc.). La détection manuelle de ces incidents est :
- **Coûteuse** : Nécessite une surveillance humaine constante
- **Lente** : Délai entre l'incident et sa détection
- **Erreur humaine** : Risque de manquer des incidents importants

### 1.2 Objectifs

1. **Automatiser la détection** : Identifier automatiquement les incidents dans les communications
2. **Réduire les faux négatifs** : Minimiser le risque de ne pas détecter un incident réel
3. **Adapter au contexte métier** : Prendre en compte les facteurs de risque spécifiques au transport médical

---

## 2. Pipeline NLP Complet

### 2.1 Collecte et Préparation des Données

#### Extraction depuis MongoDB

Les conversations ont été extraites depuis la base MongoDB en utilisant des scripts Python (via la librairie `pymongo`) :
- Identification des conversations contenant des incidents grâce aux champs internes (`incident`, `incident_report`, `not_incident`)
- Nettoyage des données : suppression des messages automatiques, des doublons et des textes trop courts
- Export dans un format JSONL structuré, prêt pour l'entraînement

#### Format des Données

Les données sont au format **JSONL** (JSON Lines), où chaque ligne représente un exemple. Les messages d'une même conversation sont **concaténés en un bloc unique**.

#### Split Train/Test

- **Train** : 8,123 exemples
- **Test** : 2,031 exemples
- **Distribution** : Déséquilibre conservé (beaucoup plus de non-incidents que d'incidents) pour refléter la réalité et simuler les conditions de production

#### Encodage des Labels

```python
label2id = {
    "non_incident": 0,
    "incident": 1
}
```

### 2.2 Préprocessing

#### Tokenisation avec CamemBERT

```python
from transformers import CamembertTokenizer

tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

def tokenize_function(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )
```

**Choix techniques** :
- **Padding** : `max_length` pour uniformiser les séquences
- **Truncation** : Limite à 128 tokens
- **Distribution réelle** : Médiane ≈ 38 tokens, 75e centile ≈ 69 tokens, max ≈ 1097 tokens
- **Justification** : 128 tokens couvre la grande majorité des conversations tout en limitant le temps de calcul

### 2.3 Modèle : CamemBERT

#### Architecture

- **Modèle de base** : `camembert-base` (Hugging Face)
- **Architecture** : Transformer BERT adapté au français
- **Paramètres** : ~110M de paramètres
- **Vocabulaire** : 32,000 tokens (SentencePiece)

#### Adaptation pour la Classification

```python
from transformers import CamembertForSequenceClassification

model = CamembertForSequenceClassification.from_pretrained(
    "camembert-base",
    num_labels=2  # Classification binaire
)
```

Le modèle ajoute une couche de classification linéaire :
- **Input** : Représentation du [CLS] token (768 dimensions)
- **Output** : 2 logits (non_incident, incident)

### 2.4 Entraînement

#### Hyperparamètres

```python
TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,  # Limitation mémoire GPU
    num_train_epochs=2,
    weight_decay=0.01,               # Pour éviter l'overfitting
    # Scheduler cosine avec warmup
)
```

#### Justifications des Hyperparamètres

- **Learning rate 2e-5** : Standard pour le fine-tuning de BERT
- **Batch size 16** : Limitation mémoire GPU (Google Colab)
- **2 epochs** : Assure une convergence rapide
- **Weight decay 0.01** : Pour éviter l'overfitting
- **Scheduler cosine avec warmup** : Optimisation de l'apprentissage

#### Infrastructure d'Entraînement

- **Plateforme** : Google Colab (accès gratuit aux GPU)
- **Durée** : Entraînements sur plusieurs heures
- **Flexibilité** : Permet de tester différents paramètres

#### Métriques d'Évaluation

```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}
```

### 2.5 Résultats d'Entraînement

![Résultats d'entraînement](../assets/training_results.png)

#### Métriques d'Entraînement

| Epoch | Training Loss | Validation Loss | Accuracy | F1-Score (Weighted) |
|-------|---------------|-----------------|----------|---------------------|
| 1     | 0.396         | 0.281           | 0.903    | 0.894               |
| 2     | 0.276         | 0.260           | 0.909    | 0.906               |

**Détails techniques** :
- **Durée totale** : 5h 41min 50s
- **Nombre de steps** : 1,016
- **Training samples/second** : 0.791

#### Performance sur le Test Set (Seuil Standard 0.5)

- **Accuracy** : ≈ 90%
- **F1-Score Global** : 0.91
- **F1-Score Incidents** : 0.73

**Analyse** :
- ✅ **Accuracy globale** : ≈ 90% (excellente)
- ✅ **F1-Score global** : 0.91 (très bon)
- ⚠️ **F1-Score incidents** : 0.73 (acceptable mais perfectible)
- ⚠️ **Faux négatifs** : La matrice de confusion a montré que le modèle produisait encore des faux négatifs (incidents réels non détectés)

---

## 3. Optimisation du Seuil de Classification

Voir [THRESHOLD_OPTIMIZATION.md](THRESHOLD_OPTIMIZATION.md) pour les détails complets.

### 3.1 Problématique du Seuil Standard

Avec un seuil fixe à 0.5 :
- **Trop de faux négatifs** (incidents réels non détectés)
- **Risque métier** : Un incident non détecté peut avoir des conséquences graves

### 3.2 Solution 1 : Seuil Optimal Fixe (0.90)

Sur demande du tuteur, l'impact du seuil de décision a été étudié :
- Extraction des probabilités prédites pour la classe incident
- Variation du seuil de 0.1 à 0.9
- Tracé des courbes précision – rappel – F1

**Résultat** : Un seuil optimal ≈ **0.90** a été identifié, qui :
- Maximise le rappel (détection des incidents)
- Maintient une précision acceptable
- Réduit significativement les faux négatifs

### 3.3 Solution 2 : Personnalisation Dynamique (Expérimentée)

Une personnalisation dynamique du seuil a également été expérimentée, en fonction de paramètres de risque identifiés :
- Type de trajet
- Contexte week-end/jour férié
- Timing des messages

Cette approche a permis de réduire fortement les faux négatifs, tout en gardant les faux positifs sous contrôle.

---

## 4. Évaluation et Validation

### 4.1 Métriques Utilisées

- **Accuracy** : Performance globale
- **Precision** : Fiabilité des prédictions positives
- **Recall** : Capacité à détecter tous les incidents
- **F1-Score** : Moyenne harmonique precision/recall
- **Matrice de confusion** : Visualisation des erreurs

### 4.2 Focus sur le Recall

Dans ce contexte métier, **le recall est plus important que la precision** :
- **Faux négatif** : Incident non détecté → **Risque critique**
- **Faux positif** : Alerte sur un non-incident → Vérification manuelle (acceptable)

### 4.3 Validation Métier

Le seuil personnalisé a été validé avec les experts métier pour :
- ✅ Réduire drastiquement les faux négatifs
- ✅ Maintenir une accuracy globale acceptable
- ✅ Adapter le système aux contraintes opérationnelles

---

## 5. Limitations et Améliorations Futures

### 5.1 Limitations Actuelles

- **Dataset** : Taille limitée (8K train, 2K test)
- **Déséquilibre** : Probable déséquilibre de classes (à préciser)
- **Features métier** : Intégration manuelle des facteurs de risque
- **Seuil fixe** : Réduction de 0.05 par critère (pourrait être optimisée)

### 5.2 Améliorations Possibles

1. **Augmentation des données** :
   - Data augmentation (paraphrasing, back-translation)
   - Collecte de plus d'exemples d'incidents

2. **Optimisation du seuil** :
   - Apprentissage automatique des poids par critère
   - Seuil adaptatif selon la distribution des probabilités

3. **Features additionnelles** :
   - Sentiment analysis
   - Entités nommées (lieux, heures, noms)
   - Historique du transporteur

4. **Modèles alternatifs** :
   - CamemBERT-large (plus de paramètres)
   - Modèles spécialisés domaine médical
   - Ensemble de modèles

---

## 6. Conclusion

Cette méthodologie présente un pipeline NLP complet pour la détection d'incidents, avec une innovation majeure : **l'adaptation du seuil de classification au contexte métier**. Cette approche permet de réduire drastiquement les faux négatifs tout en maintenant une performance globale élevée.

---

*Document basé sur le rapport de stage et les notebooks d'expérimentation*
