1. `tm_holdout_4.csv` - 4/6 antibodies used in the holdout of Abmelt for each of the 3 models are public antibodies. This data we got from `AbMelt/public_tm/holdout_internal_plus_jain.csv`
2. All `_normalized` are copied over from `AbMelt/data/`
3. All `_denormalized` are made using `denormalize_all.py` using training values from the paper, for the publically available antibodies from (1).
