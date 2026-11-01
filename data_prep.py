"""Laboratorium 3: wczytywanie i podzial danych do klasyfikacji.

Modul mozna importowac bez uruchamiania treningu i zapisu plikow.
"""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

# Plik konfiguracyjny jest odczytywany niezaleznie od katalogu uruchomienia.
PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")


def load_data(path):
    """Wczytaj dane CSV do obiektu pandas.DataFrame."""
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Zbior danych jest pusty.")
    return df


def split_data(df, test_size, seed):
    """Podziel dane na X_train, X_test, y_train, y_test.

    Stratification zapewnia podobny udzial klas w obu zbiorach.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Oczekiwano niepustego DataFrame.")
    if not 0 < test_size < 1:
        raise ValueError("TEST_SIZE musi nalezec do przedzialu (0, 1).")

    target_column = os.environ["TARGET_COLUMN"]
    if target_column not in df.columns:
        raise ValueError(f"Brak kolumny etykiet: {target_column}")

    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
