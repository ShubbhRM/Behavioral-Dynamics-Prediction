# Behavioral Dynamics Prediction with Attention-Based Neural Point Processes

This project adapts and implements a deep learning framework inspired by the paper [Modeling Behavioral Dynamics in Digital Content Consumption](https://pubsonline.informs.org/doi/abs/10.1287/mksc.2020.0180) to **predict forum thread engagement behavior**.

We address three core questions about thread activity:
- **π(t):** Will there be a reply? (Binary classification)
- **λ(t):** When will the reply occur? (Time regression)
- **count(t):** How many replies will follow in the next 24 hours? (Multi-class classification)

---

## Project Goals
- Apply point process modeling using GRU and attention to asynchronous, user-generated forum data
- Capture complex temporal and behavioral dependencies
- Evaluate all three predictive tasks in a multitask framework
- Interpret model outputs using per-thread visualizations

---

## Repository Structure

```bash
.
├── data/                             # Original CSVs (not uploaded here)
│   ├── sentiment_replies.csv
│   └── topic.csv
│
├── notebooks/                        # Full modeling pipeline
│   ├── 1_preprocessing_and_baseline_training.ipynb
│   ├── 2_feature_expansion_and_prediction_refinement.ipynb
│   └── 3_prediction_analysis_and_visualization.ipynb
│
├── models/                           # Trained models and checkpoints
│   └── best_multitask_model.pt
│
├── predictions.csv                   # Output from inference step
├── README.md                         # You're here!
└── report/                           # Final project report (optional folder)
```

---

## Notebook Summaries

### [`1_preprocessing_and_baseline_training.ipynb`](notebooks/1_preprocessing_and_baseline_training.ipynb)
- Cleans raw forum data
- Generates structured + SBERT features
- Builds and trains a multitask GRU model with decayed attention
- Outputs model checkpoint + baseline metrics

### [`2_feature_expansion_and_prediction_refinement.ipynb`](notebooks/2_feature_expansion_and_prediction_refinement.ipynb)
- Adds richer behavioral/temporal features
- Applies log transformation on λ(t)
- Buckets count(t) into 4 balanced classes
- Produces padded sequences for multitask training

### [`3_prediction_analysis_and_visualization.ipynb`](notebooks/3_prediction_analysis_and_visualization.ipynb)
- Loads saved model and predicts on validation set
- Extracts and saves per-step predictions for π(t), λ(t), and count(t)
- Generates side-by-side plots of predicted vs actual behavior

---

## Core Model Highlights

- **GRU backbone:** Encodes thread reply sequences
- **Decayed attention:** Models influence of past events for each prediction head
- **Heads:**
  - `Sigmoid` for π(t)
  - `Softplus` for λ(t)
  - `Softmax` for count(t)
- **Loss composition:**
```python
Total Loss = 2.0 * BCE(π) + 1.0 * MSE(log(λ)) + 0.5 * CE(count)
```

---

## Inspired By

> Amini et al. (2020). *Modeling Behavioral Dynamics in Digital Content Consumption*. Marketing Science.

This implementation translates their multi-attention neural point process to the domain of **online forum activity prediction**, preserving task-specific attention mechanisms and multitask learning.

---

## Example Outputs

- `predictions.csv` with model predictions per step
- Plots comparing predicted vs. actual:
  - Reply probability
  - Time until reply
  - Number of replies

---

## Future Directions

| Task                      | Motivation                              |
|---------------------------|------------------------------------------|
| Bi-GRU or Transformers    | Capture richer forward–backward context |
| Focal loss for π(t)       | Improve robustness on imbalanced classes|
| Class-weighted CE for count(t) | Address dominance of "0 reply" class |
| Use raw text threads as input | End-to-end language + behavior modeling|

---

## Final Thoughts

This repo implements a research-grade behavioral modeling pipeline end-to-end — from preprocessing, multitask prediction, to post hoc analysis — with interpretability and real-world asynchronous data in mind.

Happy modeling!
