# Overbooking pipeline v4: EUR 300 once per walked guest

Keep overbooking_pipeline_v2.py, _v3.py and _v4.py and the outputs/ and outputs_v3/ folders together in one folder, with hotels.csv.

```
python overbooking_pipeline_v4.py tune  --data hotels.csv --outdir outputs_v4 --refA outputs --refC outputs_v3   # ~2 min
python overbooking_pipeline_v4.py score --data hotels.csv --outdir outputs_v4 --refA outputs --refC outputs_v3   # ~1 min
python overbooking_pipeline_v3.py table --data hotels.csv --outdir outputs_v3 --asof 2017-08-15 --variant C      # morning table
```
Finding: re-tuning on per-stay costs overfit validation and broke the 10 per 1,000 standard at City in 2017 (19-23). Ship model C with the v3 settings (5.0 / 3.8 per 1,000, net EUR 88k vs no overselling). 2017 is exploratory.
