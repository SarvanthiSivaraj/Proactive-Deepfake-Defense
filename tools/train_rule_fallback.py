import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder


def main():
    path = os.path.join("metadata", "variant_metrics_agg.csv")
    if not os.path.exists(path):
        print("Aggregated metrics not found. Run tools/auto_tune_rules.py first.")
        return

    df = pd.read_csv(path)
    # Use numeric features
    drop_cols = [c for c in df.columns if c in ("source", "seed", "attack", "expected")]
    X = df.drop(columns=drop_cols)
    y = df["expected"].astype(str)

    # Simple label encoding
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    print(f"Fallback decision-tree test accuracy: {acc:.3f}")

    os.makedirs("metadata", exist_ok=True)
    model_path = os.path.join("metadata", "rule_fallback.joblib")
    pkg = {"model": clf, "le": le, "feature_names": list(X.columns)}
    joblib.dump(pkg, model_path)
    print(f"Saved fallback model to {model_path}")


if __name__ == "__main__":
    main()
