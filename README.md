# MetroPT-3 Predictive Maintenance: Four-Model Failure Prediction

Predicting compressor failures from industrial sensor data. The notebook compares four classifiers on a **chronologically split** dataset with severe class imbalance, and evaluates them with metrics suited to rare events (PR-AUC, precision, recall, F1).

> **Status:** part of a manuscript in preparation, *"A Machine Learning Framework for Intelligent Anomaly Detection and Predictive Monitoring of Industrial Systems"*.

## Overview

| Item | Detail |
|---|---|
| Dataset | MetroPT-3 (compressor sensor data), 1,516,948 rows |
| Features | 15 original sensors plus 28 rolling-statistical and change-based features, 43 predictors in total |
| Split | Chronological: 856,832 training rows, 660,116 test rows (18,937 failure events in the test set) |
| Models | Logistic Regression, Random Forest, XGBoost, LightGBM |
| Imbalance | `scale_pos_weight = 76.7736` for LightGBM, class weights for the other models |
| Metrics | Accuracy, precision, recall, F1-score, ROC-AUC, PR-AUC |

## Key points of the method

- **No shuffling.** The time series is split at a fixed point so the models are tested on later data than they were trained on.
- **No test-set leakage in preprocessing.** The median imputer is fitted on the training set only.
- **Feature-count guard.** The notebook stops if the input does not have exactly 43 predictors, which prevents training on an index column or on the target.
- **PR-AUC first.** With about 2.9% failures, accuracy is high for any model that predicts "normal" most of the time, so PR-AUC is the main comparison metric.
- **Threshold robustness check.** An extra section chooses each model's threshold on a validation slice from the end of the training period, then applies it once to the test set.

## Reported results (from the manuscript)

| Model | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.50 | 0.7353 | 0.7103 | 0.7226 | 0.9605 | 0.5304 |
| Random Forest | 0.03 | 0.4911 | 0.5430 | 0.5158 | 0.8861 | 0.4620 |
| XGBoost | 0.02 | 0.6546 | 0.2548 | 0.3668 | 0.9687 | 0.4717 |
| LightGBM | 0.01 | 0.7845 | 0.4330 | 0.5580 | 0.9714 | **0.6042** |

LightGBM had the best PR-AUC, ROC-AUC and precision. Logistic Regression had the best recall and F1 at its threshold.

These values come from the original experiment. Running the notebook trains the models again with the settings documented in the manuscript, and prints the difference from the reported values. A fresh run can differ slightly.

## Repository structure

```
.
├── LICENSE
├── README.md
├── requirements.txt
└── notebooks/
    └── metropt3_four_model_comparison.ipynb
```

## Run it

1. Get the MetroPT-3 dataset (search for "MetroPT-3" in the UCI Machine Learning Repository).
2. Open `notebooks/metropt3_four_model_comparison.ipynb` in Google Colab.
3. Upload the final 43-feature CSV (file name containing `final_43_features`) when asked. The raw-data feature-engineering code and the engineered file are not included in this repository.
4. Run the cells in order.

## Limitations

- One chronological split, so the results describe a single period.
- No hyperparameter search. The model settings were chosen manually as reasonable starting points.
- The thresholds were chosen to maximise F1 in the original experiment. The validation-slice check in the notebook shows how much this choice matters.
- Confidence intervals and a lead-time analysis are not included yet.

## Citation

Dataset: Veloso, B., Ribeiro, R. P., Gama, J., & Pereira, P. M. (2022). The MetroPT dataset for predictive maintenance. *Scientific Data*, 9, 764.

## License

MIT, see [LICENSE](LICENSE).
