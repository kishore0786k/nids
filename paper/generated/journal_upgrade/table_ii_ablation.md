| Config                                                 | F1_macro | MCC    | Unknown_pct | Severity_acc |
| ------------------------------------------------------ | -------- | ------ | ----------- | ------------ |
| A) XGBoost only, no SMOTE, no threshold                | 0.7482   | 0.9370 |             |              |
| B) XGBoost + SMOTE, no threshold                       | 0.7449   | 0.9289 |             |              |
| C) XGBoost + Isolation Forest, no confidence threshold | 0.6577   | 0.8770 | 4.9000      | 0.0858       |
| D) Full hybrid + unknown threshold + severity          | 0.6682   | 0.8666 | 8.9750      | 0.0858       |
