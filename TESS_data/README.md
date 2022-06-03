TESS Light curve data in light_curves_fausnaugh.zip
column headings are as follows:

```
BTJD: Barycentric TESS Julian Date = BJD - 2457000.0
TJD: TESS Julian Date = JD - 2457000.0
cts: Counts, related to brightness or flux (Often Photon count)
e_cts: Uncertainty in Counts measurement
bkg: Background Counts
bkg_model: Background Model Counts
bkg2: Second Background measurement from Fausnaugh
e_bkg2: Second Background Counts
```



sn_counts and AT_counts list transients. sn_counts contain only confirmed supernovae.
Columns are as follows.
```
sector
ra
dec
magnitude at discovery
time of discovery in TESS JD
classification
IAU name
discovery survey
cam
ccd
column
row
```