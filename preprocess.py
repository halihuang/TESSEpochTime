import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import extinction
from astroquery.ipac.irsa.irsa_dust import IrsaDust
import astropy.coordinates as coord
import astropy.units as u

bin_interval = "1D"
transients = pd.read_csv("./TESS_data/AT_count_transients_s1-47 (4).txt", names=["sector", "ra", "dec", "mag", "TJD_discovery", "type" ,"class", "IAU", "survey", "cam", "ccd", "col", "row"], delim_whitespace=True)


def preprocess(filename, display=False):
    """
    processes data of single lightcurve
    :param filename: lightcurve filename string
    :param display: whether to show plot
    :return: processed light curve in PandaDataframe
    """
    # lightcurve data
    curve = pd.read_csv("./TESS_data/light_curves_fausnaugh/" + filename, delim_whitespace=True)
    curve_name = filename.split("_")[1]
    # information about transient
    curve_meta = transients[transients['IAU'] == curve_name]
    if curve_meta.empty:
        return None, None

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
    ra = curve_meta["ra"].iloc[0]
    dec = curve_meta["dec"].iloc[0]
    flux_in = curve["cts"]
    fluxerr_in = curve["e_cts"]
    bandpass_wavelengths = np.array([786.5,])

    # Get Milky Way E(B-V) Extinction
    coo = coord.SkyCoord(ra * u.deg, dec * u.deg, frame='icrs')
    dust = IrsaDust.get_query_table(coo, section='ebv')
    mwebv = dust['ext SandF mean'][0]

    # Remove extinction from light curves
    # (Using negative a_v so that extinction.apply works in reverse and removes the extinction)
    extinction_per_passband = extinction.fitzpatrick99(wave=bandpass_wavelengths, a_v=-3.1 * mwebv, r_v=3.1, unit='aa')
    flux_out = extinction.apply(extinction_per_passband[0], flux_in, inplace=False)
    fluxerr_out = extinction.apply(extinction_per_passband[0], fluxerr_in, inplace=False)

    curve['cts'] = flux_out
    curve['e_cts'] = fluxerr_out

    # convert time to relative to discovery
    curve['relative_time'] = curve['TJD'] - curve_meta["TJD_discovery"].iloc[0]

    # bin
    # square e_cts to get variances
    curve['e_cts'] = np.power(curve['e_cts'], 2)
    # find avg cnts and avg variances
    curve.index = pd.TimedeltaIndex(curve['relative_time'].round(), unit="D")
    curve = curve.resample(bin_interval).mean()
    # sqrt avg vars to get uncertainty in stds
    curve['e_cts'] = np.power(curve['e_cts'], 0.5)

    # repalce NANs as 0s
    curve = curve.fillna(0)

    if display:
        plot_title = f"{curve_name}\n Class: {curve_meta['class'].iloc[0]}, Sector: {curve_meta['sector'].iloc[0]} \nCoords:{curve_meta['ra'].iloc[0], curve_meta['dec'].iloc[0]}, \nDiscovery TJD: {curve_meta['TJD_discovery'].iloc[0]}, Survey: {curve_meta['survey'].iloc[0]}"
        ax = curve.plot.scatter(x="relative_time", y='cts', c="00000", alpha=0.5, yerr='e_cts', ylabel="Flux", xlabel="Days relative to discovery", title=plot_title)

    return curve, curve_meta


def process_all_curves():
    """
    :return: iterator over all processed light_curves (PandaDataFrames) and their respective max light (TJD str)
    """
    light_curves = os.listdir("./TESS_data/light_curves_fausnaugh")
    i = 0
    while i < len(light_curves):
        light_curve, meta = preprocess(light_curves[i])
        if light_curve is not None:
            yield light_curve, meta
        i += 1


def save_processed_curves():
    for curve, meta in process_all_curves():
        df = pd.DataFrame({"cts": curve["cts"], "e_cts": curve["e_cts"]}, index=curve.index)
        df.index = df.index.astype("timedelta64[D]")
        filename = f"lc_{meta['IAU'].iloc[0]}_processed.csv"
        directory = "./TESS_data/processed_curves/"
        df.to_csv(directory + filename)


# save all curves as CSVs
save_processed_curves()