import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path

def train():
    # Load Data
    BASE_DIR = Path(__file__).resolve().parent.parent

    train_path = BASE_DIR / "Membangun_model" / "student_performance_preprocessing" / "train.csv"
    test_path  = BASE_DIR / "Membangun_model" / "student_performance_preprocessing" / "test.csv"

    print("Loading Data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop(columns=['final_grade'])
    y_train = train_df['final_grade']
    X_test = test_df.drop(columns=['final_grade'])
    y_test = test_df['final_grade']

    # Setup MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("CI_Experiment")
    
    # Aktifkan Autolog
    mlflow.sklearn.autolog()

    # Training
    print("Training Model...")
    with mlflow.start_run(run_name="CI_Experiment") as run:
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        # Prediksi
        y_train_pred = clf.predict(X_train)
        y_test_pred  = clf.predict(X_test)

        # Akurasi
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc  = accuracy_score(y_test, y_test_pred)

        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test  Accuracy: {test_acc:.4f}")

        # Log ke MLflow
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.sklearn.log_model(clf, "model")

        # Print Run ID agar bisa ditangkap oleh GitHub Actions
        print(f"::set-output name=run_id::{run.info.run_id}")

    print("Selesai! Cek folder 'mlruns' untuk hasil tracking lokal.")

if __name__ == "__main__":
    train()