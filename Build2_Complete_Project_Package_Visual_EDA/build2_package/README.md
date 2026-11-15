# Build 2 — How many rooms can each hotel oversell tonight?

## Files
- `build2_analysis.py`: reproducible end-to-end retrospective prototype.
- `Build2_Python_Walkthrough.ipynb`: annotated beginner-friendly workflow.
- `Build2_Final_Presentation.pptx`: 10-slide presentation with results and limitations.
- `Build2_Viva_Guide.md`: questions and answers, role-wise, plus 10-minute speaker script.
- `Problem_and_Method_Explained.md`: detailed business framing and mathematical explanation.
- `results/`: audit, full validation grid, frozen parameters, 2017 test comparisons and nightly decisions.
- `run_log.txt`: saved stdout from a full successful run.

## Run
Put `hotels.csv` in the parent directory of this package (as in the provided layout), or point to any CSV path.

```bash
python -m pip install -r requirements.txt
python build2_analysis.py --csv ../hotels.csv --out results --city-capacity 150 --resort-capacity 100
```

The sample capacities **150 city / 100 resort are hypothetical demonstration inputs, NOT facts from the dataset or brief**. Change them for sensitivity analysis. The training/validation/test dates remain fixed.

## Data split
- Train: July 2015–June 2016 (49,157 rows)
- Select: July–December 2016 (29,546 rows)
- Evaluate only after freezing: January–August 2017 (40,687 rows)

## Fundamental limitation
This CSV is a published retrospective reservation-outcome table. It does not contain actual room capacity, unique booking identifiers, as-of-night reservation snapshots, timestamped cancellation events, full room occupancy schedules, or an explicit client flat rule. Consequently none of the computed euro savings are validated operational savings. The mock evaluation is a **scenario** with guaranteed-arrival hypothetical added bookings, arrival-night-only accounting, and a cap on nights already at or above assumed booked capacity. Treat this as a defensible *proof of concept*, not an approved deployment.

## Policy definitions
- `zero`: no extra sales.
- `flat`: same integer extras on eligible nights; validation selects k.
- `historical_rate`: historical hotel-specific cancellation rate -> predicted shows; safety buffer selected in validation.
- `logistic`: scikit-learn Logistic Regression predicts show probability per reservation using booking-plausible features; adds up predictions and uses a validation-selected safety buffer.

All policies add at most 20 hypothetical, guaranteed-to-show reservations per eligible arrival night. Eligible means original arrival-date reservation count >= the assumed capacity. This is a convenience for the retrospective simulator and is NOT a realistic booking engine. Walks per 1,000 arriving modeled guests is computed on all days in the evaluation period; clarify client definitions before deployment.

## Stronger next iteration
Obtain audited capacity by date and room type, anonymized booking ID, booking and cancellation timestamps, booking-state snapshots at a fixed decision cutoff, length-of-stay inventory, guest-walking policy by segment, realistic price per room-night, and the client's reference flat rule. Use a real sequential inventory simulator and stress tests; only then authorize live rules.


## Visual EDA notebook (12 reproducible charts)
Open `Build2_Python_Walkthrough_Visual_EDA.ipynb` from the project folder and run all cells. It saves 12 labeled PNG figures to `figures/`, including cancellation patterns, lead time, ADR, deposit/segment, correlation, ROC, calibration, euro cost, walkaway ceiling and validation buffer tuning. Install matplotlib and seaborn from requirements. The notebook needs `hotels.csv` one directory above the package.
