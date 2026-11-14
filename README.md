# MetroPT-3 Predictive Maintenance: Four-Model Failure Prediction

Predicting compressor failures using industrial sensor data. In this notebook, four classification models are evaluated on a **chronologically divided** data set that exhibits an extreme level of class imbalance, using performance measures appropriate for imbalanced classes (PR-AUC, precision, recall, F1 score).

> **Status:** in preparation for publication, *"A Machine Learning Framework for Intelligent Anomaly Detection and Predictive Monitoring of Industrial Systems"*.

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

- **No shuffling.** The time series is split at a fixed point such that the models are evaluated on newer data than they were trained on.
- **No test-set leakage in preprocessing.** The median imputer is trained on the training set only.
- **Feature-count guard.** If the number of features in the input is not 43, the notebook will not proceed with the training of the model on the index column or target.
- **PR-AUC first.** Given about 2.9% failure rate, any model that is trained to predict "normal" in most cases will be accurate, thus PR-AUC is the primary metric used.
- **Threshold robustness test.** Additionally, there is a separate part where each model's threshold is selected using the validation slice at the end of the training period, and applied to the test set once.

## Reported results (from the manuscript)

| Model | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.50 | 0.7353 | 0.7103 | 0.7226 | 0.9605 | 0.5304 |
| Random Forest | 0.03 | 0.4911 | 0.5430 | 0.5158 | 0.8861 | 0.4620 |
| XGBoost | 0.02 | 0.6546 | 0.2548 | 0.3668 | 0.9687 | 0.4717 |
| LightGBM | 0.01 | 0.7845 | 0.4330 | 0.5580 | 0.9714 | **0.6042** |

LightGBM had the highest PR-AUC, ROC-AUC, and precision. Logistic Regression had the best recall and F1 scores at its threshold.

This is based on the results of the initial experiment. Running the notebook reruns the experiment with parameters as mentioned in the manuscript, and outputs the difference from the reported results. A new run may have slight differences.

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

1. Download the MetroPT-3 dataset (use “MetroPT-3” in the UCI Machine Learning Repository).
2. Click on `notebooks/metropt3_four_model_comparison.ipynb` inside Google Colab.
3. Upload the CSV file with the final 43 features (file name should have `final_43_features`) when prompted. The raw data and feature engineering code are not part of this repo.
4. Execute the cells sequentially.

## Limitations

- Chronological split done, thus results refer to a single period of time.
- No hyperparameter tuning was done. Model parameters were selected manually.
- Thresholds were selected so that F1 would be maximum in the initial experiment. Validation-slice check is demonstrated in the notebook for showing how important it was.
- Lead-time analysis and confidence intervals are not implemented yet.

## Citation

Dataset: Veloso, B., Ribeiro, R. P., Gama, J., & Pereira, P. M. (2022). The MetroPT dataset for predictive maintenance. *Scientific Data*, 9, 764.

## License

MIT, see [LICENSE](LICENSE).
