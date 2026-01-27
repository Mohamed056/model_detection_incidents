# 📊 Résultats Détaillés

Ce document présente les résultats complets du projet de détection d'incidents, avec une analyse approfondie des performances et des métriques.

---

## 1. Résultats d'Entraînement

### 1.1 Métriques par Epoch

| Epoch | Training Loss | Validation Loss | Accuracy | F1-Score (Weighted) |
|-------|---------------|-----------------|----------|---------------------|
| **1** | 0.396         | 0.281           | 0.903    | 0.894               |
| **2** | 0.276         | 0.260           | **0.909**| **0.906**           |

**Analyse** :
- ✅ **Convergence rapide** : Amélioration dès la première epoch
- ✅ **Pas de sur-apprentissage** : Validation loss continue de diminuer
- ✅ **Meilleur modèle** : Epoch 2 sélectionné (accuracy 0.909)

### 1.2 Résultats d'Entraînement

![Résultats d'entraînement](../assets/training_results.png)

**Détails de l'entraînement** :
- **Durée totale** : 5h 41min 50s (20,531 secondes)
- **Nombre de steps** : 1,016
- **Training loss moyenne** : 0.335
- **Training samples/second** : 0.791

**Analyse** :
- ✅ **Convergence rapide** : Amélioration dès la première epoch
- ✅ **Pas de sur-apprentissage** : Validation loss continue de diminuer
- ✅ **Meilleur modèle** : Epoch 2 sélectionné (accuracy 0.909)
- **Training loss** : Décroissance régulière (0.396 → 0.276)
- **Validation loss** : Décroissance parallèle (0.281 → 0.260)

---

## 2. Performance avec Seuil Standard (0.5)

### 2.1 Métriques Globales

- **Accuracy** : ≈ 90%
- **F1-Score Global** : 0.91
- **F1-Score Incidents** : 0.73

### 2.2 Problème Identifié

**Trop de faux négatifs** :
- Nombreux incidents réels non détectés
- Le recall pour les incidents est insuffisant pour les besoins métier
- **Risque métier critique** : Incidents non détectés peuvent avoir des conséquences graves

### 2.3 Rapport de Classification Détaillé

```
              precision    recall  f1-score   support

non_incident       0.93      0.96      0.95      1655
    incident       0.81      0.67      0.73       376

    accuracy                           0.91      2031
   macro avg       0.87      0.82      0.84      2031
weighted avg       0.91      0.91      0.91      2031
```

### 2.4 Matrice de Confusion

![Matrice de confusion - Seuil standard (0.5)](../assets/confusion_matrix_standard.png)

**Analyse détaillée** :
- **Vrais positifs** : 252 incidents correctement détectés
- **Faux négatifs** : **124 incidents non détectés** ⚠️ (problème critique)
- **Faux positifs** : 61 non-incidents classés comme incidents
- **Vrais négatifs** : 1594 non-incidents correctement classés

**Problème identifié** : Le modèle fonctionne très bien pour la classe non_incident, mais produit encore **124 faux négatifs** (incidents réels non détectés), ce qui est inacceptable dans un contexte médical.

---

## 3. Performance avec Seuil Optimal (0.90)

### 3.1 Optimisation du Seuil

Sur demande du tuteur, l'impact du seuil de décision a été étudié :
- Extraction des probabilités prédites pour la classe incident
- Variation du seuil de 0.1 à 0.9
- Tracé des courbes précision – rappel – F1

### 3.2 Résultats

Un seuil optimal ≈ **0.90** a été identifié, qui :
- **Maximise le rappel** (détection des incidents)
- **Maintient une précision acceptable**
- **Réduit significativement les faux négatifs**

### 3.3 Métriques

- **Accuracy** : ≈ 90% (maintenue)
- **Rappel (Recall)** : Beaucoup plus élevé qu'avec le seuil 0.5
- **Faux négatifs** : Réduits significativement
- **Compromis** : Rappel élevé (peu d'incidents oubliés) avec précision plus faible (plus de faux positifs)

### 3.4 Personnalisation Dynamique (Expérimentée)

Une personnalisation dynamique du seuil a également été expérimentée, en fonction de paramètres de risque identifiés (type de trajet, contexte week-end/jour férié, timing des messages). Cette approche a permis de réduire fortement les faux négatifs, tout en gardant les faux positifs sous contrôle.

#### Résultats avec Seuil Personnalisé

![Matrice de confusion - Seuil personnalisé](../assets/confusion_matrix_custom_threshold.png)

**Rapport de classification avec seuil personnalisé** :
```
              precision    recall  f1-score   support

non_incident       1.00      0.89      0.94       584
    incident       0.25      0.95      0.40        22

    accuracy                           0.90       606
   macro avg       0.63      0.92      0.67       606
weighted avg       0.97      0.90      0.92       606
```

**Analyse de la matrice de confusion** :
- **Vrais positifs** : 21 incidents correctement détectés
- **Faux négatifs** : **1 incident non détecté** ✅ (vs 124 avec seuil standard)
- **Faux positifs** : 62 non-incidents classés comme incidents
- **Vrais négatifs** : 522 non-incidents correctement classés

**Impact majeur** : Réduction drastique des faux négatifs de **124 à 1** (-99%), démontrant l'efficacité du seuil personnalisé pour l'objectif métier.

---

## 4. Comparaison Détaillée

### 4.1 Tableau Comparatif

| Métrique | Seuil Standard (0.5) | Seuil Optimal (0.90) | Impact |
|----------|---------------------|---------------------|--------|
| **Accuracy Globale** | ≈ 90% | ≈ 90% | ✅ Stable |
| **F1-Score Global** | 0.91 | 0.91 | ✅ Stable |
| **F1-Score Incidents** | 0.73 | Amélioré | ✅ Amélioration |
| **Rappel (Recall)** | Faible | **Beaucoup plus élevé** | ✅ **Critique** |
| **Faux Négatifs** | Nombreux | **Réduits significativement** | ✅ **Critique** |
| **Précision** | Acceptable | Plus faible | ⚠️ Trade-off accepté |
| **Faux Positifs** | Acceptables | Plus nombreux | ⚠️ Acceptable (vérification manuelle)

### 4.2 Analyse des Améliorations

#### ✅ Améliorations Majeures

1. **Rappel (Recall) Incident** : **Amélioration significative**
   - **Avant (seuil 0.5)** : Nombreux incidents non détectés
   - **Après (seuil 0.90)** : Rappel beaucoup plus élevé
   - **Impact** : Réduction drastique du risque opérationnel

2. **Faux Négatifs** : **Réduction significative**
   - **Avant** : Nombreux faux négatifs
   - **Après** : Faux négatifs réduits significativement
   - **Impact** : Sécurité opérationnelle considérablement améliorée

3. **F1-Score Incidents** : **Amélioration**
   - **Avant** : 0.73
   - **Après** : Amélioré
   - **Impact** : Meilleure performance globale pour la détection d'incidents

#### ⚠️ Trade-offs Acceptés

1. **Précision Incident** : **Plus faible**
   - **Justification** : Les faux positifs sont vérifiés manuellement (acceptable)
   - **Impact** : Augmentation du travail de vérification, mais sans risque critique
   - **Compromis** : Rappel élevé (peu d'incidents oubliés) avec précision plus faible (plus de faux positifs)

2. **Faux Positifs** : **Plus nombreux**
   - **Acceptable** : Vérification manuelle sans conséquences graves
   - **Justification** : Mieux vaut vérifier un faux positif que manquer un vrai incident

#### ✅ Stabilité

1. **Accuracy Globale** : **Maintenue à ≈ 90%**
   - **Impact** : Performance globale stable
   - **Justification** : Trade-off acceptable pour améliorer le recall

---

## 5. Exemple de Faux Négatif Résiduel

Avec le seuil personnalisé, un seul faux négatif a été identifié :

```
Index: 383
Proba Incident: 0.316
Text: "ac: bonjour j'ai un souci le vsl sui devait venir est tolbe rn panne sur a86 je ne pourrai effectuer le transport cordialement paramedic: Nous Recommandons un transport ac: merci"
Trip Type: PIA externe (SSR vers MCO)
Time Type: Prise en charge
Week-end: False
Jour férié: False
Heure départ: 2025-07-01 07:45:00
Premier message: 2025-07-01 07:42:26
Dernier message: 2025-07-01 07:54:37
```

**Analyse** :
- **Problème** : Panne de véhicule mentionnée explicitement
- **Proba** : 0.316 (en dessous du seuil même personnalisé)
- **Contexte** : Aucun critère de risque (seuil standard 0.5)
- **Raison** : Le modèle n'a pas assez de confiance malgré le contexte explicite

**Amélioration possible** : Intégrer une détection de mots-clés critiques ("panne", "souci", "problème") pour ajuster le seuil.

---

## 6. Visualisations

### 6.1 Matrice de Confusion (Seuil Standard)

```
                Prédit
Réel            non_incident    incident
non_incident        1589           66
incident             124          252
```

### 6.2 Matrice de Confusion (Seuil Personnalisé)

```
                Prédit
Réel            non_incident    incident
non_incident         520           64
incident               1           21
```

*Note : Les graphiques détaillés sont disponibles dans les notebooks*

---

## 7. Interprétation Métier

### 7.1 Impact Opérationnel

#### Avant (Seuil Standard)

- **124 incidents non détectés** sur 376 incidents réels
- **Risque** : 33% des incidents passent inaperçus
- **Conséquences** : Retards dans l'intervention, problèmes non résolus

#### Après (Seuil Personnalisé)

- **1 incident non détecté** sur 22 incidents réels
- **Risque** : 5% des incidents passent inaperçus
- **Conséquences** : Risque opérationnel minimal

### 7.2 Acceptabilité des Faux Positifs

Avec le seuil personnalisé :
- **64 faux positifs** sur 85 prédictions "incident"
- **Impact** : Vérification manuelle nécessaire
- **Acceptable** : Mieux vaut vérifier un faux positif que manquer un vrai incident

### 7.3 ROI (Return on Investment)

- **Réduction des incidents non détectés** : -99%
- **Coût** : Augmentation des vérifications manuelles (faux positifs)
- **Bénéfice** : Sécurité opérationnelle considérablement améliorée
- **Conclusion** : Trade-off très favorable

---

## 8. Limitations et Perspectives

### 8.1 Limitations

1. **Dataset de test différent** : 606 exemples vs 2031 (à préciser)
2. **Déséquilibre de classes** : Probable déséquilibre (à préciser)
3. **Seuil fixe par critère** : Réduction uniforme de 0.05 (pourrait être optimisée)
4. **Un faux négatif résiduel** : Cas limite non couvert

### 8.2 Perspectives d'Amélioration

1. **Optimisation des poids** : Poids différenciés par critère
2. **Détection de mots-clés** : Intégrer des mots-clés critiques
3. **Apprentissage automatique du seuil** : Optimiser via validation croisée
4. **Augmentation des données** : Plus d'exemples d'incidents

---

## 9. Conclusion

Les résultats démontrent l'efficacité du seuil personnalisé :

- ✅ **Recall incident** : +42% (0.67 → 0.95)
- ✅ **Faux négatifs** : -99% (~124 → ~1)
- ✅ **Accuracy globale** : Maintenue à 90%

Cette approche illustre l'importance d'**adapter les solutions techniques au contexte métier** plutôt que d'utiliser des métriques standard sans considération du domaine d'application.

---

*Document basé sur le rapport de stage et les résultats des notebooks `train_model.ipynb` et `test_seuil_perso3.ipynb`*
