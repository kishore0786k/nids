from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except:
    HAS_XGB = False


def get_models():
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
    }

    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    return models
