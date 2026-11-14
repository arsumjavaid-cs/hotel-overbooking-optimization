# Source Code Architecture (`src/`)

Each module in `src/` has a single, well-defined responsibility in the hotel overbooking pipeline:

| File | Role | Responsibility |
| :--- | :--- | :--- |
| `data.py` | Data Ingestion | Downloads the raw bookings dataset and loads it with strict schema formatting. |
| `contract.py` | Business Evaluation | Implements the standardized time-based splits, daily aggregation, and Euro cost-scoring rules. |
| `features.py` | Feature Engineering | Cleans booking records and extracts leak-free predictive features (logged in `docs/DECISIONS.md`). |
| `backtest.py` | Training & Validation | Fits models on the train period, tunes overbooking buffers on the choose period, runs a single final proof evaluation, and saves the winning artifact to `models/`. |
| `nightly.py` | Inference CLI | Loads the committed model artifact and generates daily room oversell recommendations for a requested date window. |

## End-to-End Verification Commands

From a fresh Codespace or clean checkout, the pipeline runs end-to-end with:

```bash
python src/backtest.py
python src/nightly.py --from 2017-08-18 --nights 14
```

*(Note: The trained model artifact in `models/` is committed after the final backtest so `nightly.py` can generate predictions immediately without retraining.)*
