import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    rng = np.random.default_rng(42)
    n = 300
    df = pd.DataFrame({
        "study_hours": rng.uniform(0, 10, n),
        "attendance_pct": rng.uniform(40, 100, n),
        "previous_score": rng.uniform(20, 100, n),
        "assignments_completed_pct": rng.uniform(0, 100, n),
    })

    # Synthetic demonstration label, not a real educational outcome.
    latent_score = (
        4.5 * df["study_hours"]
        + 0.20 * df["attendance_pct"]
        + 0.35 * df["previous_score"]
        + 0.08 * df["assignments_completed_pct"]
    )
    df["result"] = np.where(latent_score >= 53, "Pass", "Fail")
    df.loc[[5, 18], "attendance_pct"] = np.nan

    print("\\nDATA PREVIEW\\n", df.head())
    print("\\nMissing values:\\n", df.isna().sum())
    print("\\nSummary statistics:\\n", df.describe())

    features = ["study_hours", "attendance_pct", "previous_score",
                "assignments_completed_pct"]
    X, y = df[features], df["result"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=150, random_state=42,
                               class_weight="balanced")
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("\\nTEST ACCURACY:", round(accuracy_score(y_test, pred), 3))
    print("\\nCLASSIFICATION REPORT\\n",
          classification_report(y_test, pred, zero_division=0))
    print("CONFUSION MATRIX (Fail, Pass)\\n",
          confusion_matrix(y_test, pred, labels=["Fail", "Pass"]))

    sample = pd.DataFrame([{
        "study_hours": 6.0, "attendance_pct": 85.0,
        "previous_score": 72.0, "assignments_completed_pct": 90.0
    }])
    print("\\nNEW STUDENT PREDICTION:", model.predict(sample)[0])

    df["result"].value_counts().reindex(["Pass", "Fail"]).plot(
        kind="bar", title="Synthetic Student Result Distribution"
    )
    plt.xlabel("Result")
    plt.ylabel("Number of records")
    plt.tight_layout()
    plt.savefig("student_result_distribution.png", dpi=150)
    plt.show()
    df.to_csv("student_performance_demo.csv", index=False)
    print("\\nSaved student_performance_demo.csv and student_result_distribution.png")

if __name__ == "__main__":
    main()
