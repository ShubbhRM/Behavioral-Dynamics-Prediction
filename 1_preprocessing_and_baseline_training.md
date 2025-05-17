##  Notebook: `1_preprocessing_and_baseline_training.ipynb`

###  Purpose

This notebook initiates the **end-to-end pipeline** for behavioral prediction modeling using forum thread data. It performs:

- Data cleaning and feature engineering from raw `sentiment_replies.csv` and `topic.csv`
- Target label generation for the binary task of predicting reply occurrence (π(t))
- Text embedding using SBERT
- Structured feature normalization and fusion
- Sequence padding and preparation
- Baseline model definition using GRU + decayed attention
- Training with early stopping, loss tracking, and performance metrics
- Visual inspection of predictions vs. ground truth

---

###  Detailed Breakdown

#### 1. **Data Loading & Cleaning**
- Loads and merges `sentiment_replies.csv` and `topic.csv` using `connecting_id`
- Fixes:
  - Inconsistent datetime strings
  - Commas in numeric fields (`postcount`)
  - Missing or malformed reply indices

#### 2. **Feature Engineering**
- Computes `delta_t`: hours since previous reply
- Creates binary `label` for reply within 24 hours (π(t))
- Extracts numerical features like:
  - `reply_index`, `compound`, `views`, `replies`, etc.

#### 3. **SBERT Embedding**
- Uses `all-MiniLM-L6-v2` SentenceTransformer to embed post text (`replyMess`)
- SBERT embeddings are 384-dimensional
- Applies `MinMaxScaler` to normalize structured features

#### 4. **Data Formatting**
- Concatenates structured features and SBERT embeddings into full input vectors
- Groups by `connecting_id` to create thread-level sequences
- Applies sequence padding to allow for batch training

#### 5. **Model Architecture**
Defines the `BehavioralPointProcessModel`:
- GRU-based sequence encoder
- Three **decayed attention heads**:
  - For π(t), λ(t), and count(t)
- Task-specific output heads:
  - **Sigmoid** → reply occurrence (π(t))
  - **Softplus** → time-to-reply (λ(t))
  - **Softmax** → reply count class (count(t))
- Multi-task loss:
  ```
  Total Loss = 2.0 × reply_loss + 1.0 × lambda_loss + 0.5 × count_loss
  ```

#### 6. **Training Loop**
- 80/20 train-validation split
- Optimizer: `AdamW`
- Scheduler: `ReduceLROnPlateau`
- Logging per epoch:
  - Training and validation loss
  - Precision, recall, F1 for π(t)
  - Mean Absolute Error (MAE) for λ(t)

#### 7. **Visualizations**
- Side-by-side plots for predicted vs. true values:
  - π(t) → reply probability
  - λ(t) → time-to-next-reply
  - count(t) → number of replies in next 24 hours

---

###  Issues Encountered & Fixes

| Problem              | Description                                                                 | Fix                                   |
|----------------------|-----------------------------------------------------------------------------|----------------------------------------|
| Extreme λ(t) values  | Large gaps between replies skew regression loss                             | Applied `np.log1p()` transform         |
| Flat π(t) predictions| Model stuck around 0.5 due to class imbalance                               | Weighted BCE loss, re-checked label balance |
| Imbalanced count(t)  | Always predicted class 0 (no reply) due to skew                             | Used class bucketing + attention head  |

---

###  Notebook Output

- Final processed tensors:
  - `padded_inputs`, `padded_labels`, `padded_lambda`, `padded_count_class`, `padded_times`
- Trained model checkpoint:
  - `best_multitask_model.pt`
- Evaluation metrics (F1, MAE) on validation set
- Prediction vs. truth plots + attention heatmaps
