"""Testy jednostkowe funkcji z modulu data_prep.py."""

import os
from pathlib import Path

import pandas as pd
import pytest

from data_prep import load_data, split_data

DATA = Path(__file__).resolve().parents[1] / "data" / "wine.csv"


@pytest.fixture(autouse=True)
def target_column(monkeypatch):
    # Testy sa niezalezne od lokalnego pliku .env (takze w GitHub Actions).
    monkeypatch.setenv("TARGET_COLUMN", "target")


def test_load_data_zwraca_dataframe():
    df = load_data(DATA)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 178


def test_split_data_rozmiar_testu():
    df = load_data(DATA)
    X_train, X_test, y_train, y_test = split_data(df, 0.2, 42)
    assert len(X_train) == 142
    assert len(X_test) == 36
    assert len(y_test) == 36


def test_split_data_zachowuje_wszystkie_rekordy():
    df = load_data(DATA)
    X_train, X_test, y_train, y_test = split_data(df, 0.2, 42)
    assert len(X_train) + len(X_test) == len(df)
    assert len(y_train) + len(y_test) == len(df)
    assert set(X_train.index).isdisjoint(set(X_test.index))


def test_split_data_jest_powtarzalny():
    df = load_data(DATA)
    result_1 = split_data(df, 0.2, 42)
    result_2 = split_data(df, 0.2, 42)
    assert all(a.equals(b) for a, b in zip(result_1, result_2))


def test_split_data_pusty_dataframe():
    with pytest.raises(ValueError, match="niepustego DataFrame"):
        split_data(pd.DataFrame(), 0.2, 42)


def test_split_data_brak_kolumny_target():
    df = load_data(DATA).drop(columns=["target"])
    with pytest.raises(ValueError, match="Brak kolumny etykiet"):
        split_data(df, 0.2, 42)


def test_split_data_nieprawidlowa_proporcja():
    df = load_data(DATA)
    with pytest.raises(ValueError, match="TEST_SIZE"):
        split_data(df, 1.0, 42)
