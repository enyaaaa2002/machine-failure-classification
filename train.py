from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

root = Path(__file__).resolve().parent
csv_file = root / "data" / "ai4i2020.csv"
output = root / "artifacts_maxFeature4_leaf5"

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
FEATURES = ["Type", *NUMERIC_FEATURES]
TARGET = "Machine failure"


def load_data():
    df = pd.read_csv(csv_file)
    df = df[FEATURES + [TARGET]].copy()
    # print(df.shape)

    return df



def main():
    output.mkdir(exist_ok=True)
    df = load_data()

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessing = ColumnTransformer(
        [("category", OneHotEncoder(handle_unknown="ignore"), ["Type"])],
        remainder="passthrough"
    )

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            min_samples_leaf=5,
            max_features=4,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("模型準確率：", accuracy_score(y_test, y_pred))
    print("故障召回率：", recall_score(y_test, y_pred, zero_division=0))
    print("Average Precision：", average_precision_score(y_test, y_prob))
    print("\n分類報告：")
    print(classification_report(y_test, y_pred, zero_division=0))

    counts = df[TARGET].value_counts().reindex([0, 1], fill_value=0)
    plt.figure(figsize=(6, 4))
    plt.bar(["Normal", "Failure"], counts.values)
    plt.title("Class Distribution")
    plt.ylabel("Samples")
    plt.tight_layout()
    plt.savefig(output / "01_class_distribution.png")
    plt.close()

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Normal", "Failure"]
    )
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output / "02_confusion_matrix.png")
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_test, y_prob)
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(output / "03_precision_recall_curve.png")
    plt.close()

    importance = permutation_importance(
        model, X_test, y_test,
        scoring="average_precision",
        n_repeats=3,
        random_state=42
    )
    pd.Series(
        importance.importances_mean,
        index=FEATURES
    ).sort_values().plot.barh(figsize=(8, 5))
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(output / "04_feature_importance.png")
    plt.close()

    joblib.dump(model, output / "model.joblib")

    demo = X_test.copy()
    demo["observed_label"] = y_test.to_numpy()
    demo["predicted_probability"] = y_prob

    low = demo["predicted_probability"].idxmin()
    high = demo["predicted_probability"].idxmax()

    demo.loc[[low, high]].to_csv(
        output / "demo_examples.csv", index=False
    )

    # print("\n模型訓練完成！")
    # print("圖表與模型已儲存至 artifacts 資料夾")


if __name__ == "__main__":
    main()
