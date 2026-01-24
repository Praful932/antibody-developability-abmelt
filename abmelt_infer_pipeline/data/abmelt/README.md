Data for which we need to reproduce metrics as reported in the paper
| Endpoint                      | R²          | r_p²          |
| ----------------------------- | ----------- | ------------- |
| Tagg(Aggregation Temperature) | 0.57 ± 0.11 | 0.71 ± 0.09   |
| Tm,on(Melting Onset)          | 0.56 ± 0.01 | 0.61 ± 0.0003 |
| Tm(Melting Temperature)       | 0.60 ± 0.06 | 0.64 ± 0.04   |

1. `tm_holdout_4.csv` - 4/6 antibodies used in the holdout of Abmelt for each of the 3 models are public antibodies. This data we got from `AbMelt/public_tm/holdout_internal_plus_jain.csv`
2. All `_normalized` are copied over from `AbMelt/data/`
3. All `_denormalized` are made using `denormalize_all.py` using training values from the paper, for the publically available antibodies from (1).

Note : `tm_holdout_4` is the denormalized version from the actual dataset used and tm_holdout_denormalized is the derived denormalized version. Code used is `src/denormalize_all.py`.
