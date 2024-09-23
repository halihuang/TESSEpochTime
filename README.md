# TESS-Epoch-Time
predicts time until max light and and time until explosion of TESS transients 


## Preprocessing: 
```python
#place AT_counts file and unzipped light_curves_fausnaugh in TESS_data/
python preprocessing.py
```

## Training
```python
#place unzipped preprocessed_curves into TESS_data/ or run preprocessing step above
python training.py
```
