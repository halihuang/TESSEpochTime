import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import extinction
from astroquery.ipac.irsa.irsa_dust import IrsaDust
import astropy.coordinates as coord
import astropy.units as u

#config params
time_scale = "first" # {"first", "trigger", "BTJD", "TJD"}, determines whether to index relative to first observation or trigger, BTJD, or TJD time"
to_bin = True
bin_interval = "0.5D" # Day: D, Minute: T, Second: S
transients = pd.read_csv("./TESS_data/AT_count_transients_s1-47 (4).txt", names=["sector", "ra", "dec", "mag", "TJD_discovery", "type" ,"class", "IAU", "survey", "cam", "ccd", "col", "row"], delim_whitespace=True)


def bin_curves(df, interval, time_col="relative_time"):
    """
    :param df: panda dataframe with cts and e_cts cols
    :param interval: str, scalar + unit, eg. 0.5D, Units: Day: D, Minute: T, Second: S
    :param time_col: label of col of time
    :return:
    """
    binned = df.copy()
    # square e_cts to get variances
    binned['e_cts'] = np.power(binned['e_cts'], 2)
    # bin and find avg variance and avg mean per bin
    binned.index = pd.TimedeltaIndex(df[time_col], unit="D").round(interval)
    binned = binned.resample(interval, origin="start").mean()
    binned.index = binned.index / pd.to_timedelta(interval)
    # sqrt avg vars to get uncertainty in stds
    binned['e_cts'] = np.power(binned['e_cts'], 0.5)
    return binned


def preprocess(filename, display=False):
    """
    processes data of single lightcurve
    :param filename: lightcurve filename string
    :param display: whether to show plot
    :param relative:
    :return: processed light curve in PandaDataframe
    """
    # lightcurve data
    original = pd.read_csv("./TESS_data/light_curves_fausnaugh/" + filename, delim_whitespace=True)
    curve_name = filename.split("_")[1]
    curve = original.copy()
    # information about transient
    curve_meta = transients[transients['IAU'] == curve_name]
    if curve_meta.empty:
        return None, None, None
    curve_meta = curve_meta.iloc[0]

    # sigma clipping
    for _ in range(0, 5):
        uncert_mean = curve.e_cts.mean()
        threshold = 3*curve.e_cts.std()
        curve = curve[np.abs(curve['e_cts'] - uncert_mean) <= threshold]

    # sub bg flux
    if not curve['bkg_model'].isnull().all():
        curve['cts'] = curve['cts'] - curve['bkg_model']

    # correct milky way extinction
    # Set relevant parameters
    ra = curve_meta["ra"]
    dec = curve_meta["dec"]
    flux_in = curve["cts"]
    fluxerr_in = curve["e_cts"]
    bandpass_wavelengths = np.array([7865,])

    # Get Milky Way E(B-V) Extinction
    coo = coord.SkyCoord(ra * u.deg, dec * u.deg, frame='icrs')
    b = coo.galactic.b.value
    dust = IrsaDust.get_query_table(coo, section='ebv')
    mwebv = dust['ext SandF mean'][0]

    # Remove extinction from light curves
    # (Using negative a_v so that extinction.apply works in reverse and removes the extinction)
    extinction_per_passband = extinction.fitzpatrick99(wave=bandpass_wavelengths, a_v=-3.1 * mwebv, r_v=3.1, unit='aa')
    flux_out = extinction.apply(extinction_per_passband[0], flux_in, inplace=False)
    fluxerr_out = extinction.apply(extinction_per_passband[0], fluxerr_in, inplace=False)

    curve['cts'] = flux_out
    curve['e_cts'] = fluxerr_out
    curve_meta["mwebv"] = mwebv
    curve_meta["gal_lat"] = b

    if time_scale == "trigger":
        # convert time to relative to discovery
        curve['relative_time'] = curve['TJD'] - curve_meta["TJD_discovery"]

    if time_scale == "first":
        # convert time to relative to 1st observation
        curve['relative_time'] = curve['BTJD'] - curve['BTJD'].iloc[0]

    if time_scale == "BTJD":
        curve['relative_time'] = curve['BTJD']

    if time_scale == "TJD":
        curve['relative_time'] = curve['TJD']

    # bin
    if to_bin:
        curve = bin_curves(curve, bin_interval)
    else:
        curve.index = curve['relative_time']

    # replace NANs as 0s
    curve = curve.fillna(0)

    if display:
        plot_title = f"{curve_name}\n Class: {curve_meta['class']}, Sector: {curve_meta['sector']} \nCoords:{curve_meta['ra'], curve_meta['dec']}, \nDiscovery TJD: {curve_meta['TJD_discovery']}, Survey: {curve_meta['survey']}"
        ax = curve.plot.scatter(x="relative_time", y='cts', c="00000", alpha=0.5, yerr='e_cts', ylabel="Flux", xlabel="Days relative to discovery", title=plot_title)

    return curve, curve_meta, original


def process_all_curves():
    """
    :return: iterator over all processed light_curves (PandaDataFrames) and their respective max light (TJD str)
    """
    light_curves = os.listdir("./TESS_data/light_curves_fausnaugh")
    i = 0
    while i < len(light_curves):
        light_curve, meta, original = preprocess(light_curves[i])
        if light_curve is not None:
            yield light_curve, meta, original
        i += 1


def save_processed_curves(directory):
    outliers = []
    mwebv_outliers = []

    for curve, meta, original in process_all_curves():
        df = pd.DataFrame({"cts": curve["cts"], "e_cts": curve["e_cts"]}, index=curve.index)
        curve_id = meta['IAU']
        filename = f"lc_{curve_id}_processed.csv"
        if meta['mwebv'] > 0.5:
            mwebv_outliers.append({"mwebv": meta['mwebv'], "class": meta['class'], 'gal_lat': meta['gal_lat'], 'IAU': curve_id})
        else:
            if df['cts'].max() > 10 * original['cts'].max():
                outliers.append(curve_id)
            df.to_csv(directory + filename)

    with open("./TESS_data/potential_outliers.txt", "w") as outlier_file:
        outlier_file.write(str(outliers))

    mwebv_df = pd.DataFrame(mwebv_outliers)
    mwebv_df.to_csv("./TESS_data/mwebv_outliers.csv", index=False)



# preprocess("lc_2018fbm_cleaned")

# save all curves as CSVs
save_processed_curves("./TESS_data/processed_curves/")

