# SMILES-2026 Hallucination Detection Solution

## Reproducibility

To reproduce the run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python solution.py
```

This command extracts hidden states from `Qwen/Qwen2.5-0.5B`, builds the feature
matrix, evaluates the probe, writes `results.json`, and creates
`predictions.csv`.

The run used `USE_GEOMETRIC = True`. The generated `predictions.csv` contains
100 test examples:

| Predicted label | Count |
|---:|---:|
| 0 | 18 |
| 1 | 82 |

## Results

The table reports the values from `results.json`.

| Metric | Value |
|---|---:|
| Number of labelled samples | 689 |
| Number of folds | 5 |
| Split size per fold | 447-448 train / 104 val / 137-138 test |
| Feature dimension | 48,789 |
| Feature extraction time | 11.18 s |
| Baseline accuracy | 70.10% |
| Baseline F1 | 82.42% |
| Train accuracy | 89.04% |
| Train F1 | 92.63% |
| Train AUROC | 96.49% |
| Validation accuracy | 74.04% |
| Validation F1 | 82.76% |
| Validation AUROC | 76.57% |
| Test accuracy | 74.60% |
| Test F1 | 83.03% |
| Test AUROC | 75.28% |

The test accuracy across folds ranges from 72.46% to 77.54%.

## Method

Only the three allowed files were changed: `aggregation.py`, `probe.py`, and
`splitting.py`.

In `aggregation.py`, hidden states are used from layers 4, 8, 12, 16, 20, and
the last layer. For each layer, several pooled representations are
concatenated: the last token, means over the last 4/8/16/32 tokens, the full
mean, a recency-weighted mean, and a few difference vectors. These features are
meant to emphasize the generated answer, especially the end of the response.

Scalar geometric features are also added when `USE_GEOMETRIC = True`. These
include sequence length, truncation, activation norms, cosine similarities
between different pooled vectors, and layer-to-layer drift features.

In `probe.py`, a simple linear probe ensemble is used. First, all features are
standardized. Then features are ranked on the training fold using a
between-class versus within-class variance score. Regularized logistic probes
are trained on the top 64, 128, 256, 512, and 1024 ranked features. The final
score is the weighted average of these probes, with a fixed threshold of 0.45.

In `splitting.py`, stratified 5-fold evaluation is used. Inside each fold, a
stratified validation set is split from the training part. This gives train,
val, and test metrics while keeping class proportions stable.

## Experiments

The first version used a small MLP probe with dropout and AdamW. It reached a
reasonable F1 score, but the training AUROC was too high, which suggested
overfitting on the small labelled set. This direction was replaced by simpler
linear probes.

Several PCA and logistic-regression ensembles were also tried. PCA projections
gave a useful way to reduce dimensionality, but they made the probe more
complex and were not needed after feature selection was added. The final probe
therefore removes PCA and keeps only supervised top-k feature subsets.

Extra threshold tuning and out-of-fold model weighting were tested as well.
These methods made the code harder to reason about and did not give a clear
enough benefit. A fixed threshold of 0.45 was kept instead.

Training a linear probe on the full feature matrix was tried after PCA removal.
This was not useful: training became slower and the train metrics became too
high, suggesting memorization. The top-k feature subsets gave a cleaner
train/test gap and kept the probe faster.

The split strategy also changed during development. A single
train/validation/test split was replaced by stratified k-fold evaluation for
more stable local estimates. Validation was later restored inside each fold, so
the final results include train, validation, and held-out test metrics.

The final solution is intentionally simple. Most of the signal comes from
answer-focused hidden-state aggregation, while the classifier remains a
regularized linear model.
