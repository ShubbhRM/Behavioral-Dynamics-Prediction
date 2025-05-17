## 📓 Notebook: `2_feature_expansion_and_prediction_refinement.ipynb`

### 🧠 Purpose
This notebook focuses on improving the earlier baseline by expanding features, refining count prediction targets, adding new behavioral indicators, and rebalancing prediction tasks. It includes both preprocessing and advanced target engineering steps while preparing data for training a neural point process model for behavioral prediction on forum threads.

---

### 🧩 Detailed Section-by-Section Breakdown

#### 1. **Data Loading & Cleaning**
- Same two datasets used: `sentiment_replies.csv` and `topic.csv`
- Initial cleaning includes:
  - Datetime formatting
  - Postcount strings with commas
  - Extracting `reply_index`
  - Merging on `connecting_id`

#### 2. **Target Engineering Enhancements**
- Extends the baseline binary label π(t) with two new targets:
  - **λ(t):** Time to next reply (regression). `log1p` is applied to stabilize extreme values.
  - **count(t):** Number of replies in 24h (converted into 4-class classification: `0`, `1`, `2`, `≥3`)
- Introduces `bucketize_count()` helper function to simplify the learning space and address class imbalance.

#### 3. **Behavioral and Temporal Features**
Introduces domain-specific features to enhance model input:
- `step_index`: Position of post in thread
- `hour`: Hour of the day (normalized)
- `weekday`: Day of week (normalized)
- `is_thread_starter`: Boolean (whether author is the thread starter)

These augment the model with temporal and structural awareness that was missing in the previous notebook.

#### 4. **SBERT Textual Embeddings**
- Uses `all-MiniLM-L6-v2` SentenceTransformer for 384-dimensional embeddings of `replyMess`.
- Embeddings are combined with the engineered features to form a **395-dimensional input vector**.

#### 5. **Sequence Grouping & Padding**
- Groups replies by `connecting_id` (thread ID) to form thread-level sequences
- Pads sequences for:
  - Input features
  - Binary reply labels
  - Lambda targets
  - Count bucket labels
  - Time deltas

This padded output feeds into the training model, just like the baseline, but now supports full multitask learning.

---

### 🔄 Differences from `1_preprocessing_and_baseline_training.ipynb`

| Feature                   | `1_preprocessing_and_baseline_training.ipynb`     | `2_feature_expansion_and_prediction_refinement.ipynb` |
|---------------------------|---------------------------------------------------|--------------------------------------------------------|
| **Count prediction**      | Ignored or raw count                              | **Bucketized into 4 classes** (0/1/2/≥3)                |
| **Lambda target**         | Used raw time deltas                              | **Applies `log1p` for stability**                      |
| **Additional features**   | Basic sentiment + views, replies                  | **Adds hour, weekday, is_thread_starter, step index**  |
| **Model type**            | Baseline multitask                                | **Enhanced multitask with expanded inputs**            |
| **Text encoding**         | SBERT                                             | SBERT                                                  |
| **Class imbalance handling** | Minimal                                       | **Bucketized counts and rebalanced loss**              |

---

### ✅ Outcome of This Notebook

- Cleaned and engineered dataset with richer feature representation
- Prepared padded input and target tensors:
  - `padded_inputs`, `padded_labels`, `padded_lambda`, `padded_count_class`, `padded_times`
- Ready for use in **advanced multi-head model training**
