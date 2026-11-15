# Overbooking pipeline

Rooms to oversell, per hotel, for each of the next 14 nights, with a euro proof against the flat rule.

## Run it

```
python3 overbooking_pipeline.py fit   --data hotels.csv --outdir outputs   # train < 2016-07-01, choose settings on Jul-Dec 2016, freeze
python3 overbooking_pipeline.py prove --data hotels.csv --outdir outputs   # score the frozen rule ONCE on Jan-Aug 2017
python3 overbooking_pipeline.py table --data hotels.csv --outdir outputs --asof 2017-08-15
        [--capacity '{"City Hotel": 226, "Resort Hotel": 185}']            # overwrite the estimated room counts
```

Needs pandas, numpy, scipy, scikit-learn, openpyxl. `fit` takes about 40 seconds.

## What the rule does

1. For every booking still on the books on a given morning, estimate the chance it cancels or no-shows before arrival, given how many days are left (gradient boosting on bin-1 columns only).
2. Add these up per night. Choose the oversell O that minimises expected cost: EUR 300 per guest walked, plus the hotel's monthly average ADR per empty room.
3. Output: booking limit = rooms + O, and how many more bookings can still be sold.

Settings chosen on Jul-Dec 2016 only: model configuration (1 of 4), a per-hotel log-odds drift shift, and a variance multiplier (3.0). The 2017 test was run once after they were frozen.

## Bins

* Features (bin 1): hotel, lead_time, arrival month/week/day/weekday, country, market_segment, distribution_channel, is_repeated_guest, previous_cancellations, previous_bookings_not_canceled, reserved_room_type, agent, company present, customer_type, plus k (days to arrival) and age on the books.
* Excluded though bin 1: arrival_date_year (cannot extrapolate), deposit_type (failed the leakage test, 99% cancel rate for Non Refund).
* Bin 2 used only as bookkeeping, never a predictor: stay length (places a booking on its nights; the file holds the final length, ~15% of bookings were amended) and ADR (value of an empty room).
* Bin 3 used only to reconstruct what was known on a past morning, and as the label when scoring.

## Result of the one-time proof (Jan-Aug 2017)

Decisions scored: those where bookings were already at least 80% of capacity. Totals are per-night averages over the 14 daily decisions.

| Rule | EUR cost | Rooms walked | Rooms left empty |
|---|---|---|---|
| New rule | 388,320 | 1,066 | 746 |
| Flat rule as written | 5,599,847 | 18,666 | 2 |
| Flat rule, fairer version | 372,254 | 1,068 | 548 |
| No overselling | 219,215 | 0 | 2,239 |

* The new rule beats the flat rule as written by about EUR 5.2m (95% interval 4.4m to 6.0m). That comparison is not meaningful: read literally, the flat rule oversells 80+ rooms a night at City.
* Against the fairer flat rule (same formula, cancellation rate measured on bookings still standing at that horizon) it is a tie: -16k, interval -110k to +80k.
* **It does not beat selling no extra rooms, overall.** Resort: yes (57.5k vs 70.0k). City: no (330.8k vs 149.2k).
* Cause: City cancellations among bookings still standing a week out in Apr-Aug fell from 20.3% (2015) to 9.0% (2016) to 4.5% (2017). The model learned the older, higher rate.

## Assumptions to confirm

* Capacity is estimated from the occupancy plateau (City 226, Resort 185), not the real room count.
* Flat rule: hotel average cancellation rate (41.7% City, 25.2% Resort, from training) x bookings on the books, less 3 rooms. The 3 is assumed for "a few".
* Cost assumes the extra rooms get sold to bookings that behave like that night's other bookings, and enough demand exists to fill the limit. Refused demand is not in the data, so this cannot be tested.
* Walk cost EUR 300 comes from the brief and is the same for every guest.

## Before anyone relies on it

* Add rolling recalibration (re-fit the drift shift weekly from resolved bookings). It was not tried on 2017 because the one-time proof is spent; it needs fresh data to prove.
* Replace estimated capacity with the real room counts, and the flat-rule numbers with what the revenue manager actually uses.
