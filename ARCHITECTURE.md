# 🏗️ Project Architecture

This document presents the technical architecture and structure of the incident detection project.

---

## 1. Overview

```
Model paramedic/
│
├── README.md                    # Main documentation (recruiter-oriented)
├── ARCHITECTURE.md              # This file
├── .gitignore                   # Files to ignore
│
├── docs/                        # Detailed documentation
│   ├── METHODOLOGY.md           # Complete methodology
│   ├── THRESHOLD_OPTIMIZATION.md # Threshold optimization
│   └── RESULTS.md               # Detailed results
│
└── notebooks/                   # Experimentation notebooks
    ├── train_model.ipynb        # CamemBERT model training
    └── test_seuil_perso3.ipynb  # Custom threshold tests
```

---

## 2. Processing Pipeline

### 2.1 Data Flow

```
Raw data (JSONL)
    ↓
Preprocessing
    ├── Label encoding (non_incident: 0, incident: 1)
    └── Tokenization (CamemBERT, max_length=128)
    ↓
Hugging Face Dataset
    ├── Train (8,123 examples)
    └── Test (2,031 examples)
    ↓
Training
    ├── Model: CamemBERT-base
    ├── Fine-tuning: 2 epochs
    └── Optimized hyperparameters
    ↓
Trained model
    ↓
Evaluation
    ├── Standard threshold (0.5)
    └── Custom threshold (dynamic)
    ↓
Results and metrics
```

### 2.2 Main Components

#### 2.2.1 Preprocessing

- **Input format** : JSONL (one line per example)
- **Fields used** :
  - `text` : Communication message
  - `label` : Label (non_incident / incident)
  - `trip_type` : Transport type
  - `time_type` : Time type
  - `is_weekend` : Boolean
  - `is_bank_holidays` : Boolean
  - `dt_starting` : Scheduled departure time
  - `first_message_dt` : First message time
  - `latest_message_dt` : Latest message time

- **Tokenization** :
  - Tokenizer : `CamembertTokenizer`
  - Max length : 128 tokens
  - Padding : `max_length`
  - Truncation : Enabled

#### 2.2.2 Model

- **Architecture** : `CamembertForSequenceClassification`
- **Base** : `camembert-base` (Hugging Face)
- **Parameters** : ~110M
- **Output** : 2 logits (binary classification)

#### 2.2.3 Training

- **Framework** : Hugging Face Transformers (Trainer)
- **Backend** : PyTorch
- **Hyperparameters** : See [METHODOLOGY.md](docs/METHODOLOGY.md)

#### 2.2.4 Classification

- **Standard threshold** : 0.5 (default)
- **Custom threshold** : Dynamic based on context
  - Base : 0.5
  - Reduction : -0.05 per risk criterion
  - Minimum : 0.3

---

## 3. Model Architecture

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
Probabilities [P(non_incident), P(incident)]
```

### 3.2 Classification with Custom Threshold

```
Model probabilities
    ↓
P(incident) = probas[:, 1]
    ↓
Custom threshold calculation
    ├── Base threshold: 0.5
    ├── Reduction per risk criterion: -0.05
    └── Minimum threshold: 0.3
    ↓
Comparison: P(incident) > custom_threshold ?
    ↓
Final prediction
```

---

## 4. Technologies and Dependencies

### 4.1 Main Libraries

- **transformers** (Hugging Face) : Pre-trained models and fine-tuning
- **datasets** (Hugging Face) : Dataset management
- **torch** (PyTorch) : Computing backend
- **scikit-learn** : Metrics and evaluation
- **numpy** : Numerical computations
- **matplotlib/seaborn** : Visualizations

### 4.2 Versions

*To be specified according to the environment used*

- Python : 3.8+
- transformers : 4.x+
- torch : 1.x+
- scikit-learn : 1.x+

---

## 5. Data Structure

### 5.1 Input Format (JSONL)

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

### 5.2 Format After Tokenization

```python
{
  "input_ids": [5, 1234, 5678, ...],  # Encoded tokens
  "attention_mask": [1, 1, 1, ...],    # Attention mask
  "label": 1                           # Encoded label (0 or 1)
}
```

### 5.3 Output Format

```python
{
  "predictions": [[logit_0, logit_1], ...],  # Raw logits
  "label_ids": [0, 1, 0, ...],               # Real labels
  "probabilities": [[0.8, 0.2], ...]         # Probabilities (softmax)
}
```

---

## 6. Execution Flow

### 6.1 Training Phase

1. **Data loading** : Reading JSONL files
2. **Preprocessing** : Label encoding and tokenization
3. **Model initialization** : Loading `camembert-base`
4. **Training configuration** : Hyperparameters
5. **Training** : Fine-tuning over 2 epochs
6. **Saving** : Model and tokenizer saved

### 6.2 Evaluation Phase

1. **Model loading** : Trained model
2. **Test preprocessing** : Test data tokenization
3. **Prediction** : Probability generation
4. **Classification with standard threshold** : Fixed threshold at 0.5
5. **Classification with custom threshold** : Dynamic threshold
6. **Evaluation** : Metric calculation (precision, recall, F1, accuracy)
7. **Visualization** : Confusion matrices

---

## 7. Extension Points

### 7.1 Possible Improvements

1. **Production pipeline** :
   - REST API for real-time prediction
   - Integration into a monitoring system
   - Automatic alerts

2. **Optimization** :
   - Model quantization (size reduction)
   - Inference optimization (ONNX, TensorRT)
   - Prediction caching

3. **Monitoring** :
   - Production performance tracking
   - Data drift detection
   - Threshold A/B testing

4. **Model improvement** :
   - Continuous fine-tuning (online learning)
   - Model ensembles
   - Models specialized by incident type

---

## 8. Security and Confidentiality

### 8.1 Sensitive Data

- ⚠️ **No confidential data** : Examples presented are fictional
- ⚠️ **Anonymization** : No real company or client names
- ⚠️ **Compliance** : Respect of regulations (GDPR, etc.)

### 8.2 Best Practices

- **Versioning** : Git for code
- **Documentation** : Complete project documentation
- **Testing** : Validation on separate test data
- **Reproducibility** : Fixed seeds for reproducibility

---

## 9. Deployment

### 9.1 Development Environment

- **Jupyter Notebooks** : Experimentation and prototyping
- **Google Colab** : GPU training (if used)

### 9.2 Production (Perspectives)

- **REST API** : Flask/FastAPI to serve the model
- **Containerization** : Docker for isolation
- **Orchestration** : Kubernetes for scalability
- **Monitoring** : Logs and performance metrics

---

## 10. Conclusion

This architecture presents a complete and modular NLP pipeline for incident detection, with a major innovation: **dynamic adaptation of the classification threshold to business context**. The structure is designed to be extensible and maintainable, allowing future improvements.

---

*Document based on notebook analysis and project methodology*
