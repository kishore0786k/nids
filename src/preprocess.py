import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
import joblib
import os

DATA_PATH = os.path.join("data", "NF-ToN-IoT-V2.csv")
FINAL_LABEL_COL = "label"
SCALER_PATH = os.path.join("models", "scaler.pkl")
TRAIN_OUT = os.path.join("data", "train_processed.csv")
TEST_OUT = os.path.join("data", "test_processed.csv")

# Map raw Attack types to 7 final classes
LABEL_MAP = {
    "Benign": "Benign",
    "benign": "Benign",

    "dos": "DoS/DDoS",
    "ddos": "DoS/DDoS",

    "scanning": "Scanning",

    "backdoor": "Backdoor",

    "injection": "Injection",

    "password": "Password",

    "xss": "XSS/MITM",
    "mitm": "XSS/MITM",
}

def load_and_clean():
    df = pd.read_csv(DATA_PATH)

    # Adjust these names if different in your CSV
    attack_col_candidates = ["Attack", "ATTACK", "attack_type"]
    attack_col = None
    for c in attack_col_candidates:
        if c in df.columns:
            attack_col = c
            break
    if attack_col is None:
        raise ValueError("Could not find Attack column. Check your CSV header.")

    # Keep only rows with valid Attack labels
    df[attack_col] = df[attack_col].astype(str).str.strip()
    df[FINAL_LABEL_COL] = df[attack_col].map(LABEL_MAP)
    df = df.dropna(subset=[FINAL_LABEL_COL])

    # Drop non-numeric/id columns commonly present
    drop_cols = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT",
                 "L4_DST_PORT", attack_col, "Label"]
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Keep only numeric columns
    num_df = df.select_dtypes(include=["number"])
    num_df[FINAL_LABEL_COL] = df[FINAL_LABEL_COL].values

    return num_df

def preprocess_and_split(test_size=0.2, random_state=42):
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    df = load_and_clean()
    X = df.drop(columns=[FINAL_LABEL_COL])
    y = df[FINAL_LABEL_COL]

    rus = RandomUnderSampler(random_state=random_state)
    X_res, y_res = rus.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=test_size, stratify=y_res, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    joblib.dump(scaler, SCALER_PATH)

    train_df = pd.DataFrame(X_train_s, columns=X.columns)
    train_df[FINAL_LABEL_COL] = y_train.values
    test_df = pd.DataFrame(X_test_s, columns=X.columns)
    test_df[FINAL_LABEL_COL] = y_test.values

    train_df.to_csv(TRAIN_OUT, index=False)
    test_df.to_csv(TEST_OUT, index=False)

    print("Preprocessing done. Saved train_processed.csv and test_processed.csv in data/")

if __name__ == "__main__":
    preprocess_and_split()
