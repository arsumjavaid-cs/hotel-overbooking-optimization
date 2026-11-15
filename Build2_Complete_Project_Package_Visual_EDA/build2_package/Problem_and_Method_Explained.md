# Build 2 from scratch: understanding the decision

## 1. Who is the client and what do they want?
RateGain's revenue-science team must advise a hotel group consisting of a Lisbon city property and an Algarve resort. The action is to recommend **k additional rooms beyond capacity** for a particular night. The objective is to minimize euro loss subject to the client's ceiling of **10 walked guests per 1,000 arrivals**. The brief says an empty room costs that night's average booked stay value; walking an arriving guest costs €300. The business wants a working command and auditable evidence, not just a classifier.

## 2. Understand overbooking using the brief's example
For a hotel with 20 rooms and k=4, the hotel accepts 24 reservations. If 6 cancel, only 18 show and two rooms are empty. If 4 cancel, 20 arrive and the hotel fills exactly. If 2 cancel, 22 arrive and two guests must be walked. A good rule trades unused inventory against the expensive walk.

## 3. Business loss formula
For night d, hotel h and proposed k:

- Realized arriving parties A = observed staying parties + hypothetical additional arriving parties.
- Empty rooms E = max(0, capacity - A).
- Walked guests W = max(0, A - capacity).
- Illustrative loss L(h,d,k) = E × night stay value in euros + W × €300.
- Safety metric = 1,000 × total W / total arriving parties.

Important: the CSV has no capacity field. We use **arbitrary illustrative capacities** (150 City, 100 Resort), NOT real capacities. The simulator additionally assumes each hypothetical new reservation definitely shows. It counts reservation-arrival cohorts, not multi-night room inventory. Thus it is an education-oriented approximation, not an actual hotel dispatch policy.

## 4. Dataset
119,390 records and 32 original columns, with `hotel`, `is_canceled`, `lead_time`, arrival components, stay length, booking/guest segments, ADR, history and final reservation statuses. There are 31,994 rows that match another row in all provided original fields: **do not delete them automatically**; separate parties can have identical attributes, and there is no unique reservation ID. Missingness is concentrated in `company` (112,593), `agent` (16,340), `country` (488), `children` (4). Some ADR values are zero or negative and there are extreme outliers. `reservation_status` and `reservation_status_date` reveal final outcomes and must not be model inputs.

## 5. Why leakage matters
A model that predicts cancellation using `reservation_status` will appear nearly perfect because the status is the answer. `assigned_room_type`, `booking_changes`, `days_in_waiting_list`, and potentially other fields may reflect events after the prediction cutoff. The provided prototype uses a conservative set of plausibly booking-known columns. Even these require timestamp-level audit before real deployment.

## 6. Chronological split
- Train: 2015-07-01 to 2016-06-30, inclusive: 49,157 records.
- Validate/model selection: 2016-07-01 to 2016-12-31, inclusive: 29,546 records.
- Test frozen policies: 2017-01-01 to 2017-08-31, inclusive: 40,687 records.

Why? Random splitting lets future trends leak into training. Validation decides all thresholds and policy configurations. The 2017 data is strictly held out until after freezing; no retuning on 2017 is allowed.

## 7. Four policies
1. **Zero**: k=0, the don't-oversell reference.
2. **Flat**: constant additional rooms per eligible night, 0–20; choose the integer with the lowest 2016 validation euro cost under safety ceiling.
3. **Hotel historical rate**: p_cancel estimated on training separately for each hotel; estimated showing guests = booked × (1 − p_cancel). Buffer selected on validation.
4. **Logistic regression**: one row = one reservation. Given booking-plausible fields, predict each reservation's probability of **showing up**. Sum probabilities within hotel/arrival date to estimate show count. Buffer selected on validation.

The model uses imputation, scaling for numeric fields, one-hot encoding for categorical features, and LogisticRegression. Prediction AUC/Brier are diagnostics, not the rule-selection metric.

## 8. Decision formula and buffer
A simple dynamic rule estimates available inventory after cancellations:

    suggested_k = clip(round(C - predicted_shows - safety_buffer), 0, 20)

Only suggest k on arrival-date cohorts with `bookings >= C` for the demonstration. The buffer expresses caution: a larger buffer means fewer extra rooms and normally fewer walked parties. For each hotel/method, choose a parameter on the 2016 validation slice minimizing euro loss subject to at most 10 walks/1,000 arrivals. If no parameter is feasible, mark that fact rather than claiming a safe solution.

## 9. Observed 2017 results in the demonstration

### City Hotel (assumed C=150)
| Method | Frozen parameter | Total simulated cost | Walks per 1,000 | Savings vs zero |
|---|---:|---:|---:|---:|
| Zero | 0 | €2,239,283.84 | 0.000 | €0 |
| Flat | k=20 | €2,151,287.74 | 0.000 | €87,996.10 |
| Historical rate | buffer 0 | €2,162,128.94 | 0.000 | €77,154.90 |
| Logistic | buffer 0 | €2,154,683.54 | 0.000 | €84,600.30 |

### Resort Hotel (assumed C=100)
| Method | Frozen parameter | Total simulated cost | Walks per 1,000 | Savings vs zero |
|---|---:|---:|---:|---:|
| Zero | 0 | €1,580,350.37 | 1.973 | €0 |
| Flat | k=7 | €1,586,733.54 | 4.691 | **−€6,383.17** |
| Historical rate | buffer 20 | €1,580,590.07 | 2.082 | **−€239.70** |
| Logistic | buffer 20 | €1,579,506.17 | 1.970 | €844.20 |

These are **illustrative scenario calculations**, NOT actual cost savings. In particular, a zero-oversell baseline can still have walks because the hypothetical capacity is below realized attendance on some nights.

## 10. How to interpret the result
The validation-selected flat rule is lowest-cost for the Resort in 2016, but underperforms in 2017. That is precisely why honest time testing matters. A tiny apparent gain from a more complex model may not justify its maintenance cost, especially when the scenario assumptions dominate the result. The team should present both the euro metric and walk rate, against both zero and the client's actual flat rule (the latter was NOT provided, so cannot be measured).

## 11. Client's three observations
The source deck mentions 'the client's three observations' as an assessed requirement, but the three observations themselves are **not present** in the uploaded announcement. Request the original RFP or briefing; do not invent them.

## 12. Operational next steps
Obtain exact per-night hotel capacities, unique reservation IDs, status-transition timestamps, booking cutoff snapshots, average booked stay value definition, actual flat reference rule, group-level guest counts, nightly occupied room-nights across stays, room types and no-show timing. Use a true multi-night inventory simulator and scenario stress tests. Obtain review/approval before any live guest-facing use.
