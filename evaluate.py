"""Laboratorium 3: samodzielna ocena modelu bez ponownego treningu."""

import os
from pathlib import Path

import joblib
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, classification_report

from data_prep import load_data, split_data

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def main():
    # Podzial ma identyczne parametry jak w train.py.
    data_path = project_path(os.environ["DATA_PATH"])
    model_path = project_path(os.environ["MODEL_PATH"])
    results_path = project_path(os.environ["RESULTS_PATH"])
    test_size = float(os.environ["TEST_SIZE"])
    seed = int(os.environ["RANDOM_SEED"])

    df = load_data(data_path)
    _, X_test, _, y_test = split_data(df, test_size, seed)

    # Odczyt juz wytrenowanego modelu: bez wywolania fit().
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    text = (
        "LABORATORIUM 3 - OCENA MODELU\n"
        f"Liczba probek testowych: {len(X_test)}\n"
        f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n"
        "Raport klasyfikacji:\n" + report
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"Wyniki zapisano do: {results_path}")


if __name__ == "__main__":
    main()
