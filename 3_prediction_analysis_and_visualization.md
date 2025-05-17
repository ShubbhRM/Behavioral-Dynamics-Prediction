## Notebook: `3_prediction_analysis_and_visualization.ipynb`

### Purpose

This notebook focuses on **evaluating and visualizing** the trained model’s performance across three behavioral prediction tasks. It computes predicted vs actual values for:
- **π(t):** Whether a reply will occur
- **λ(t):** When it will occur (in hours)
- **count(t):** How many replies will occur in the next 24 hours

It serves as the **final validation and interpretability notebook** for the project.

---

### Detailed Section-by-Section Breakdown

#### 1. **Model Loading**
- Defines the model architecture (`BehavioralPointProcessModel`)
- Loads trained weights from `best_multitask_model.pt`

#### 2. **Prediction Generation**
- Passes validation inputs through the model
- Extracts outputs for:
  - `reply_prob` (π(t))
  - `lambda` (λ(t))
  - `count_logits` (count(t))
- Converts logits to predicted classes using `argmax`
- Applies masks to filter padded entries

#### 3. **Organizing Outputs**
- Constructs a `DataFrame` containing:
  - `thread_id`, `step`
  - `π(t)_pred`, `π(t)_true`
  - `λ(t)_pred`, `λ(t)_true`
  - `count(t)_pred`, `count(t)_true`
- Saves results as `predictions.csv`

#### 4. **Visualization**
- Generates three side-by-side line plots:
  - π(t): Reply probability
  - λ(t): Time to next reply
  - count(t): Reply volume class
- Each plot overlays predicted and true values for a single thread (user-defined `thread_idx`)

---

### Differences from Previous Notebooks

| Aspect                        | `2_feature_expansion_and_prediction_refinement`           | `3_prediction_analysis_and_visualization`                  |
|------------------------------|------------------------------------------------------------|------------------------------------------------------------|
| 📌 Purpose                   | Feature engineering, training                              | Evaluation and interpretability                           |
| 🔄 Model                     | Trained and saved                                          | Loaded from disk and applied                              |
| 📊 Output                    | Trained model + input tensors                              | Per-step predictions and visual analysis                  |
| 📁 File Output               | —                                                          | `predictions.csv`                                         |
| 📈 Visuals                   | —                                                          | π(t), λ(t), count(t) vs. ground truth                     |

---

### Outcome of This Notebook

- Structured predictions saved in `predictions.csv`
- Visual plots for thread-level model behavior:
  - Reply classification
  - Time prediction (in log-hours)
  - Count classification
- Supports both quantitative and qualitative model evaluation

---

