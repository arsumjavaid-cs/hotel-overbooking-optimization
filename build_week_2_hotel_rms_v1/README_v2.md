# Overbooking pipeline v2 (constrained)

```
python3 overbooking_pipeline_v2.py fit      --data hotels.csv --outdir outputs   # ~2 min
python3 overbooking_pipeline_v2.py prove    --data hotels.csv --outdir outputs   # EXPLORATORY, see below
python3 overbooking_pipeline_v2.py table    --data hotels.csv --outdir outputs --asof 2017-08-15
python3 overbooking_pipeline_v2.py segments --data hotels.csv --outdir outputs   # guest types for marketing
```

## What changed
1. **Walk standard is a constraint.** For each night, the oversell is the cheapest one whose expected walk rate stays under 0.35 x 10 per 1,000 arrivals (EUR 300 per walk vs the monthly ADR per empty room). The 0.35 factor, the variance multiplier (1.5) and the drift handling were chosen on Jul-Dec 2016 so that both hotels stayed under 7 per 1,000 there.
2. **Second-bin columns tried, caveat beside each.** Adopted (validation log-loss gain 0.0063, threshold 0.003): stay nights, adults, children, babies, meal, days in waiting list, ADR. Tried but never adopted: booking_changes and total_of_special_requests (they stop growing when a booking cancels, so they partly encode the outcome). The file stores final values, so a value may include changes made after the morning being predicted. That can flatter validation; live, read the value as it stands that morning.
3. **Drift.** The log-odds shift is refitted every morning from bookings that arrived in the last 91 days (outcomes known). 91 days was the longest window tried and the best on validation; a longer window was not tested.

## Second look, Jan-Aug 2017 (exploratory, not a clean proof)
v1 already used this window once, and v1's failure there motivated the constraint and the drift handling. Settings were still chosen on Jul-Dec 2016 only, but treat this as a check, not a proof. A clean proof needs data after Aug 2017.

| Rule | EUR cost | Walked per 1,000 (both / City / Resort) |
|---|---|---|
| New (constrained) | 153,579 | 3.1 / 4.3 / 1.6 |
| Same model, cost only | 157,643 | 3.6 / 4.8 / 1.9 |
| Flat, fairer | 372,254 | 15.0 / 24.2 / 2.0 |
| No overselling | 219,215 | 0 |
| Flat, as written | 5,599,847 | 208.7 / 243.0 / 156.3 |

Saving vs no overselling: EUR 65.6k (95% interval 33k to 95k). Vs fairer flat: EUR 218.7k (152k to 286k).
Hotel-level walk rate is under 10 at both hotels. City was above 10 in Jan (43.1) and Feb (22.4), months with few scored decisions; from March it ran at 0.2 to 5.9.

## Limits
* Walks are modelled: extra rooms are assumed to behave like the night's other bookings and enough demand is assumed to fill the limit.
* Capacity (City 226, Resort 185) is estimated from the occupancy plateau.
* Flat rule: training-period average cancellation rate x bookings on the books, less 3 rooms (3 assumed).
* Guest types: statistical groups (silhouette 0.19, so they overlap); names are hypotheses.
