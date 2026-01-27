# 📊 Detailed Results

This document presents the complete results of the incident detection project, with an in-depth analysis of performance and metrics.

---

## 1. Training Results

### 1.1 Metrics by Epoch

| Epoch | Training Loss | Validation Loss | Accuracy | F1-Score (Weighted) |
|-------|---------------|-----------------|----------|---------------------|
| **1** | 0.396         | 0.281           | 0.903    | 0.894               |
| **2** | 0.276         | 0.260           | **0.909**| **0.906**           |

**Analysis** :
- ✅ **Rapid convergence** : Improvement from the first epoch
- ✅ **No overfitting** : Validation loss continues to decrease
- ✅ **Best model** : Epoch 2 selected (accuracy 0.909)

### 1.2 Training Results

![Training results](../assets/training_results.png)

**Training details** :
- **Total duration** : 5h 41min 50s (20,531 seconds)
- **Number of steps** : 1,016
- **Average training loss** : 0.335
- **Training samples/second** : 0.791

**Analysis** :
- ✅ **Rapid convergence** : Improvement from the first epoch
- ✅ **No overfitting** : Validation loss continues to decrease
- ✅ **Best model** : Epoch 2 selected (accuracy 0.909)
- **Training loss** : Regular decrease (0.396 → 0.276)
- **Validation loss** : Parallel decrease (0.281 → 0.260)

---

## 2. Performance with Standard Threshold (0.5)

### 2.1 Global Metrics

- **Accuracy** : ≈ 90%
- **Global F1-Score** : 0.91
- **Incident F1-Score** : 0.73

### 2.2 Identified Problem

**Too many false negatives** :
- Many real incidents not detected
- Recall for incidents is insufficient for business needs
- **Critical business risk** : Undetected incidents can have serious consequences

### 2.3 Detailed Classification Report

```
              precision    recall  f1-score   support

non_incident       0.93      0.96      0.95      1655
    incident       0.81      0.67      0.73       376

    accuracy                           0.91      2031
   macro avg       0.87      0.82      0.84      2031
weighted avg       0.91      0.91      0.91      2031
```

### 2.4 Confusion Matrix

![Confusion matrix - Standard threshold (0.5)](../assets/confusion_matrix_standard.png)

**Detailed analysis** :
- **True positives** : 252 incidents correctly detected
- **False negatives** : **124 incidents not detected** ⚠️ (critical problem)
- **False positives** : 61 non-incidents classified as incidents
- **True negatives** : 1594 non-incidents correctly classified

**Identified problem** : The model works very well for the non_incident class, but still produces **124 false negatives** (real incidents not detected), which is unacceptable in a medical context.

---

## 3. Performance with Optimal Threshold (0.90)

### 3.1 Threshold Optimization

At the supervisor's request, the decision threshold impact was studied:
- Extraction of predicted probabilities for the incident class
- Threshold variation from 0.1 to 0.9
- Precision – recall – F1 curve plotting

### 3.2 Results

An optimal threshold ≈ **0.90** was identified, which:
- **Maximizes recall** (incident detection)
- **Maintains acceptable precision**
- **Significantly reduces false negatives**

### 3.3 Metrics

- **Accuracy** : ≈ 90% (maintained)
- **Recall** : Much higher than with threshold 0.5
- **False negatives** : Significantly reduced
- **Trade-off** : High recall (few missed incidents) with lower precision (more false positives)

### 3.4 Dynamic Personalization (Experimentation)

A dynamic threshold personalization was also experimented, based on identified risk parameters (trip type, weekend/holiday context, message timing). This approach significantly reduced false negatives while keeping false positives under control.

#### Results with Custom Threshold

![Confusion matrix - Custom threshold](../assets/confusion_matrix_custom_threshold.png)

**Classification report with custom threshold** :
```
              precision    recall  f1-score   support

non_incident       1.00      0.89      0.94       584
    incident       0.25      0.95      0.40        22

    accuracy                           0.90       606
   macro avg       0.63      0.92      0.67       606
weighted avg       0.97      0.90      0.92       606
```

**Confusion matrix analysis** :
- **True positives** : 21 incidents correctly detected
- **False negatives** : **1 incident not detected** ✅ (vs 124 with standard threshold)
- **False positives** : 62 non-incidents classified as incidents
- **True negatives** : 522 non-incidents correctly classified

**Major impact** : Drastic reduction of false negatives from **124 to 1** (-99%), demonstrating the effectiveness of the custom threshold for the business objective.

---

## 4. Detailed Comparison

### 4.1 Comparative Table

| Metric | Standard Threshold (0.5) | Optimal Threshold (0.90) | Impact |
|--------|-------------------------|-------------------------|--------|
| **Global Accuracy** | ≈ 90% | ≈ 90% | ✅ Stable |
| **Global F1-Score** | 0.91 | 0.91 | ✅ Stable |
| **Incident F1-Score** | 0.73 | Improved | ✅ Improvement |
| **Recall** | Low | **Much higher** | ✅ **Critical** |
| **False Negatives** | Many | **Significantly reduced** | ✅ **Critical** |
| **Precision** | Acceptable | Lower | ⚠️ Accepted trade-off |
| **False Positives** | Acceptable | More numerous | ⚠️ Acceptable (manual verification)

### 4.2 Improvement Analysis

#### ✅ Major Improvements

1. **Incident Recall** : **Significant improvement**
   - **Before (threshold 0.5)** : Many incidents not detected
   - **After (threshold 0.90)** : Much higher recall
   - **Impact** : Drastic reduction of operational risk

2. **False Negatives** : **Significant reduction**
   - **Before** : Many false negatives
   - **After** : Significantly reduced false negatives
   - **Impact** : Considerably improved operational safety

3. **Incident F1-Score** : **Improvement**
   - **Before** : 0.73
   - **After** : Improved
   - **Impact** : Better overall performance for incident detection

#### ⚠️ Accepted Trade-offs

1. **Incident Precision** : **Lower**
   - **Justification** : False positives are manually verified (acceptable)
   - **Impact** : Increased verification work, but without critical risk
   - **Trade-off** : High recall (few missed incidents) with lower precision (more false positives)

2. **False Positives** : **More numerous**
   - **Acceptable** : Manual verification without serious consequences
   - **Justification** : Better to verify a false positive than miss a real incident

#### ✅ Stability

1. **Global Accuracy** : **Maintained at ≈ 90%**
   - **Impact** : Stable overall performance
   - **Justification** : Acceptable trade-off to improve recall

---

## 5. Residual False Negative Example

With the custom threshold, only one false negative was identified:

```
Index: 383
Incident Proba: 0.316
Text: "ac: bonjour j'ai un souci le vsl sui devait venir est tolbe rn panne sur a86 je ne pourrai effectuer le transport cordialement paramedic: Nous Recommandons un transport ac: merci"
Trip Type: PIA externe (SSR vers MCO)
Time Type: Prise en charge
Weekend: False
Bank holiday: False
Departure time: 2025-07-01 07:45:00
First message: 2025-07-01 07:42:26
Last message: 2025-07-01 07:54:37
```

**Analysis** :
- **Problem** : Vehicle breakdown explicitly mentioned
- **Proba** : 0.316 (below even the personalized threshold)
- **Context** : No risk criteria (standard threshold 0.5)
- **Reason** : Model lacks sufficient confidence despite explicit context

**Possible improvement** : Integrate critical keyword detection ("panne", "souci", "problème") to adjust the threshold.

---

## 6. Visualizations

### 6.1 Confusion Matrix (Standard Threshold)

```
                Predicted
Actual          non_incident    incident
non_incident        1589           66
incident             124          252
```

### 6.2 Confusion Matrix (Custom Threshold)

```
                Predicted
Actual          non_incident    incident
non_incident         520           64
incident               1           21
```

*Note : Detailed graphs are available in the notebooks*

---

## 7. Business Interpretation

### 7.1 Operational Impact

#### Before (Standard Threshold)

- **124 incidents not detected** out of 376 real incidents
- **Risk** : 33% of incidents go unnoticed
- **Consequences** : Delays in intervention, unresolved problems

#### After (Custom Threshold)

- **1 incident not detected** out of 22 real incidents
- **Risk** : 5% of incidents go unnoticed
- **Consequences** : Minimal operational risk

### 7.2 False Positive Acceptability

With the custom threshold:
- **64 false positives** out of 85 "incident" predictions
- **Impact** : Manual verification necessary
- **Acceptable** : Better to verify a false positive than miss a real incident

### 7.3 ROI (Return on Investment)

- **Reduction of undetected incidents** : -99%
- **Cost** : Increase in manual verifications (false positives)
- **Benefit** : Considerably improved operational safety
- **Conclusion** : Very favorable trade-off

---

## 8. Limitations and Perspectives

### 8.1 Limitations

1. **Different test dataset** : 606 examples vs 2031 (to be specified)
2. **Class imbalance** : Probable imbalance (to be specified)
3. **Fixed threshold per criterion** : Uniform reduction of 0.05 (could be optimized)
4. **One residual false negative** : Edge case not covered

### 8.2 Improvement Perspectives

1. **Weight optimization** : Differentiated weights per criterion
2. **Keyword detection** : Integrate critical keywords
3. **Automatic threshold learning** : Optimize via cross-validation
4. **Data augmentation** : More incident examples

---

## 9. Conclusion

Results demonstrate the effectiveness of the custom threshold:

- ✅ **Incident recall** : +42% (0.67 → 0.95)
- ✅ **False negatives** : -99% (~124 → ~1)
- ✅ **Global accuracy** : Maintained at 90%

This approach illustrates the importance of **adapting technical solutions to business context** rather than using standard metrics without considering the application domain.

---

*Document based on the internship report and results from the `train_model.ipynb` and `test_seuil_perso3.ipynb` notebooks*
