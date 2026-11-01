# IUM - Laboratorium 3: strukturyzacja pipeline'u i CI

Projekt jest kontynuacja laboratorium 2. Uzywa tego samego zbioru Wine,
`StandardScaler` i `LogisticRegression`. Siec PyTorch z laboratorium 2
nie jest tutaj potrzebna: instrukcja lab. 3 wymaga zapisu pipeline'u
scikit-learn.

## Windows / PowerShell

```powershell
cd lab03_ium_projekt
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py
python evaluate.py
python -m pytest -v
```

Jezeli PowerShell blokuje aktywacje venv, uzyj `.venv\Scripts\python.exe`
w miejsce `python` (takze przy `-m pip` i `-m pytest`).

## Struktura

- `data_prep.py`: importowalne funkcje `load_data(path)` i `split_data(df, test_size, seed)`
- `train.py`: trenowanie pipeline'u i zapis do `.joblib`
- `evaluate.py`: wczytanie gotowego modelu, klasyfikacja, zapis metryk
- `tests/test_data_prep.py`: 7 automatycznych testow
- `.env`: przykladowa, lokalna konfiguracja (ignorowana przez Git)
- `.env.example`: same nazwy wymaganych kluczy, bez wartosci
- `data/wine.csv`: te same dane Wine co w lab. 2

Uwaga: `evaluate.py` wymaga wczesniejszego uruchomienia `train.py`,
ale **nie trenuje modelu ponownie**.

## GitHub

Przed oddaniem: utworz wlasne repozytorium na GitHub, wypchnij projekt
i wklej jego adres do sprawozdania. Nie dodawaj `.env` do repozytorium.
