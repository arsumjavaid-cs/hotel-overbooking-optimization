# Hotel Overbooking Optimization Engine

A revenue-science decision system designed for a two-hotel group in Portugal to determine the optimal number of rooms to oversell each night—balancing empty-room opportunity cost against a **€300 guest-walk penalty** and a **1% maximum walk ceiling**.

![Overbooking Results](images/overbooking_performance.png)

## Tech Stack & Tools
* **Language:** Python 3.11
* **Data Manipulation:** pandas, NumPy
* **Machine Learning & Optimization:** scikit-learn, XGBoost, SciPy
* **Testing & Code Quality:** pytest
* **Visualization:** Matplotlib, Seaborn

## Business Problem & Evaluation Rules
* **Objective:** Predict daily booking cancellations and recommend an optimal nightly oversell buffer ($k$) for each property.
* **Asymmetric Cost Function:** Every unfilled room costs the night's average stay value (`ADR × nights`), while every walked guest incurs a fixed **€300 relocation penalty**.
* **Service Constraint:** Walked guests must remain at or below **10 per 1,000 arrivals (1.0%)**.
* **Chronological Validation:** Trained on bookings prior to July 2016, tuned on H2 2016, and evaluated once on the 2017 out-of-time proof period (`src/contract.py`).

## Quickstart & Reproducibility

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Download the dataset:**
   ```bash
   python src/data.py
   ```
3. **Run full backtest and verify scoring rules:**
   ```bash
   pytest -q
   python src/backtest.py
   ```
4. **Generate daily revenue-manager recommendations:**
   ```bash
   python src/nightly.py --from 2017-08-18 --nights 14
   ```

## Repository Structure
* `data/` — Dataset source documentation and download instructions (`data/README.md`).
* `notebooks/` — Exploratory data analysis, leak-free column audit, and model experiments.
* `src/` — Modular production pipeline (`data.py`, `contract.py`, `features.py`, `backtest.py`, `nightly.py`).
* `models/` — Serialized winning model artifact used for daily inference.
* `docs/` — Technical decision log (`DECISIONS.md`), feature dictionary (`COLUMNS.md`), and model comparison.

## Data Attribution
Antonio, N., de Almeida, A., & Nunes, L. (2019). *Hotel booking demand datasets*. Data in Brief, 22, 41–49. Licensed under **CC BY 4.0**. See `data/README.md` for details.
