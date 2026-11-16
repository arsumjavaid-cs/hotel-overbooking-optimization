# Overbooking pipeline v3: seasonality

Keep `overbooking_pipeline_v3.py` and `overbooking_pipeline_v2.py` in the same folder (v3 reuses v2).

```
python overbooking_pipeline_v3.py fit                    # ~4 min; compares four designs on Jul-Dec 2016, saves each
python overbooking_pipeline_v3.py prove --variant C      # scores a design on Jan-Aug 2017 (omit --variant for the validation winner)
python overbooking_pipeline_v3.py table --asof 2017-08-15 --variant C
```
Results go to `outputs_v3/` (the winner) and `outputs_v3/variant_A|B|C|E/`.

## Evidence of seasonality (arrivals before 2017)
Late cancellations (bookings still live 7 days out that later cancel or no-show): Resort 2-3% in Jun-Sep, 6-12% in Dec-Feb. City 13-16% in Jan-Mar 2016, 6-7% in Jul-Sep 2016. Summer is also when overselling matters most.
Problem: training holds one cycle (Jul 2015 - Jun 2016), and the level was drifting during it, so month and drift are tangled.

## Four designs, chosen on Jul-Dec 2016
| Design | Validation cost (EUR) | Log-loss, no validation fitting | AUC |
|---|---|---|---|
| A calendar columns in the model (= v2) | 114,040 | 0.1983 | 0.802 |
| B no calendar columns, no drift tracking | 99,750 | 0.1794 | 0.826 |
| C no calendar + fitted seasonal curve + daily drift shift | 106,335 | 0.1854 | 0.816 |
| E annual curve as model inputs | 123,712 | 0.1950 | 0.803 |

C separates booking risk, an annual curve per hotel (2 harmonics, fitted with a linear trend term and out-of-fold predictions), and a rolling level shift measured on top of the curve.
Validation picked B. Validation only covers Jul-Dec, so it cannot judge the curve for Jan-Jun.

## Third look, Jan-Aug 2017 (exploratory: v1 and v2 already used this window)
| Rule | EUR cost | Walked per 1,000 (City / Resort) |
|---|---|---|
| v2 | 153,579 | 4.3 / 1.6 |
| B (validation winner) | 172,663 | 6.4 / 0.1 |
| C (seasonal design) | 143,694 | 3.0 / 1.9 |
| No overselling | 219,215 | 0 / 0 |
| Fairer flat rule | 372,254 | 24.2 / 2.0 |

* C vs v2: saves EUR 9.9k, 95% interval -10k to +31k (cheaper in 61% of weeks). Not distinguishable.
* C vs B: saves EUR 29k (10k to 54k). B has no drift tracking and over-predicted cancellations in 2017.
* Month-by-month fit (mean |log(predicted/actual cancels)|): v2 0.29, C 0.36, B 0.49. C did NOT fit the monthly pattern better than v2.
* City worst month for C was 14.8 per 1,000 (low-volume January); hotel level is under 10.

## Reading
Seasonality is real, but one training cycle cannot separate it from drift, and none of the three ways of modelling it clearly beat v2. Treat C as the seasonal version, not a proven improvement. The fix is data: with Sep 2017 onward, refit on two full years and re-test on a window nobody has looked at.
