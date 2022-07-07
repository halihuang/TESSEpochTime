import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import extinction
from astroquery.ipac.irsa.irsa_dust import IrsaDust
import astropy.coordinates as coord
import astropy.units as u
import shutil

meta_targets = ["trigger"]


def bin_curves(df, interval, time_col="relative_time", uncert="e_cts"):
    """
    :param uncert: str for column name of uncertainty
    :param df: panda dataframe with cts and e_cts cols
    :param interval: str, scalar + unit, eg. 0.5D, Units: Day: D, Minute: T, Second: S
    :param time_col: label of col of time
    :return:
    """
    binned = df.copy()
    # square e_cts to get variances
    binned[uncert] = np.power(binned[uncert], 2)
    # bin and find avg variance and avg mean per bin
    binned.index = pd.TimedeltaIndex(df[time_col], unit="D").round(interval)
    binned = binned.resample(interval, origin="start").mean()
    binned.index = binned.index / pd.to_timedelta(interval)
    # sqrt avg vars to get uncertainty in stds
    binned[uncert] = np.power(binned[uncert], 0.5)
    return binned


def normalize(df, uncert="e_cts", light="cts"):
    e_cts = df[uncert]
    cts = df[light]
    max_cts = cts.max()
    min_cts = cts.min()
    normalized_cts = (cts - min_cts) / (max_cts - min_cts)
    normalized_ects = e_cts / (max_cts - min_cts)
    return normalized_cts, normalized_ects


def sigma_clip(df, col,  times=5, const=3):
    for _ in range(0, times):
        mean = df[col].mean()
        threshold = const * df[col].std()
        df = df[np.abs(df[col] - mean) <= threshold]
    return df


def convert_to_bin_timescale(value, interval):
    return pd.Timedelta(value, unit="D").round(interval) / pd.to_timedelta(interval)


def convert_cts_to_mag(cts, sec, is_uncert=False):
    exposure_time = 1425.6 if sec in range(1, 27) else 475.2
    zero_point = 0.0 if is_uncert else 20.44
    mag = zero_point - 2.5 * np.log10(cts / exposure_time)
    return mag


def get_curve_meta(curve_name, transients_dir="./TESS_data/AT_count_transients_s1-47 (4).txt"):
    transients = pd.read_csv(transients_dir,
                             names=["sector", "ra", "dec", "mag", "TJD_discovery", "type", "class", "IAU", "survey",
                                    "cam", "ccd", "col", "row"], delim_whitespace=True)
    curve_meta = transients[transients['IAU'] == curve_name]
    if curve_meta.empty:
        return None
    else:
        return curve_meta.iloc[0]

def preprocess_ztf(filename, parameters):
    # lightcurve data
    curve = pd.read_csv("./TESS_data/ztf_data/" + filename)
    split = filename.split("_")
    tess_curve_name = split[0]
    # convert from MJD to JD to TJD
    curve['TJD'] = curve['mjd'] + 2400000.5 - 2457000.0
    # information about transient
    curve_meta = get_curve_meta(tess_curve_name)
    # return nothing if no info found
    if curve_meta is None:
        return None, None
    curve_meta['ztf'] = split[1]
    # split into passbands and create ZTF df
    ztf_data = {}
    passbands = {'r': 2, 'g': 1}
    for pb, fid in passbands.items():
        pb_df = curve[curve['fid'] == fid]
        mag_str = f'{pb}_mag'
        uncert_str = f'{pb}_uncert'
        if not pb_df.empty:
            to_process = pd.DataFrame({"TJD": pb_df['TJD'], mag_str: pb_df['magpsf'], uncert_str: pb_df["sigmapsf"]})
            processed_pb = preprocess(to_process, curve_meta, light=mag_str, uncert=uncert_str,
                                      to_bin=parameters['to_bin'], bin_interval=parameters['bin_interval'],
                                      time_scale=parameters['time_scale'], norm=parameters['norm'])
            ztf_data[mag_str] = processed_pb[mag_str]
            ztf_data[uncert_str] = processed_pb[uncert_str]
        else:
            ztf_data[mag_str] = pd.Series(dtype='float64')
            ztf_data[uncert_str] = pd.Series(dtype='float64')

    ztf_df = pd.DataFrame(ztf_data)
    return ztf_df, curve_meta


def preprocess_tess(filename, parameters, curve_meta=None):
    original = pd.read_csv("./TESS_data/light_curves_fausnaugh/" + filename, delim_whitespace=True)
    if curve_meta is None:
        tess_curve_name = filename.split("_")[1]
        curve_meta = get_curve_meta(tess_curve_name)
        if curve_meta is None:
            return None, None
    processed = preprocess(original, curve_meta, convert_to_mag=parameters['convert_to_mag'], to_bin=parameters['to_bin'],
                           bin_interval=parameters['bin_interval'], time_scale=parameters['time_scale'],
                           norm=parameters['norm'], sub_bg_model=parameters['sub_bg_model'])

    if parameters['convert_to_mag']:
        tess_df = pd.DataFrame({"tess_mag": processed['cts'], "tess_uncert": processed['e_cts']})
    else:
        tess_df = processed
    return tess_df, curve_meta


def preprocess_ztf_tess(ztf_filename, parameters):
    params = parameters.copy()
    ztf_df, df_meta = preprocess_ztf(ztf_filename, params)
    if df_meta is None:
        return None, None

    tess_filename = f'lc_{df_meta["IAU"]}_cleaned'
    params['convert_to_mag'] = True
    params['sub_bg_model'] = True
    tess_df, _ = preprocess_tess(tess_filename, params)
    return ztf_df.join(tess_df, how="outer"), df_meta


def display_curve(light_curve, curve_meta, light, uncert, index="index", title_info=""):
    curve = light_curve.copy()
    curve['index'] = light_curve.index
    plot_title = f"{title_info}\n" \
                 f"{curve_meta['IAU']}\n Class: {curve_meta['class']}, Sector: {curve_meta['sector']} " \
                 f"\nCoords:{curve_meta['ra'], curve_meta['dec']}, \nDiscovery TJD: {curve_meta['TJD_discovery']}, " \
                 f"Survey: {curve_meta['survey']}"
    ax = curve.plot.scatter(x=index, y=light, c="00000", alpha=0.5, yerr=uncert, ylabel="Flux",
                            xlabel=f"units ({curve_meta['interval']}) relative to discovery", title=plot_title)
    return ax


def preprocess(curve, curve_meta, light="cts", uncert="e_cts", sub_bg_model=False,
               convert_to_mag=False, to_bin=True, norm=True, bin_interval="0.5D", time_scale="trigger"):
    """
    processes data of single lightcurv
    :param curve: panda dataframe containint light curve data
    :param curve_meta: meta information obtained from AT_transients
    :param light: str key for light column (cts or mag) of DataFrame
    :param uncert: str key for uncert column of Dataframe
    :param sub_bg_model: if background model flux should be subtracted from real
    :param bin_interval: Day: D, Minute: T, Second: S
    :param norm: normalizes curve
    :param to_bin: bins data according to bin interval
    :param convert_to_mag: converts cts to magnitude
    :param time_scale: one of ["first", "trigger", "BTJD", "TJD"], determines whether to index relative to first
    observation or trigger, BTJD, or TJD time"
    :return: processed light curve in PandaDataframe
    """

    if convert_to_mag:
        sector = curve_meta['sector']
        curve['cts'] = convert_cts_to_mag(curve['cts'], sector, is_uncert=False)
        curve['e_cts'] = convert_cts_to_mag(curve['e_cts'], sector, is_uncert=True)
        curve['bkg_model'] = convert_cts_to_mag(curve['bkg_model'], sector, is_uncert=False)

    if curve_meta.empty:
        return None, None
    curve_meta['interval'] = bin_interval

    # sigma clipping by e_cts
    curve = sigma_clip(curve, uncert)

    # sub bg flux
    if sub_bg_model:
        if not curve['bkg_model'].isnull().all():
            curve[light] = curve[light] - curve['bkg_model']

    # sigma clipping by cts
    ra = curve_meta["ra"]
    dec = curve_meta["dec"]
    flux_in = curve[light]
    fluxerr_in = curve[uncert]
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

    curve[light] = flux_out
    curve[uncert] = fluxerr_out
    curve_meta["mwebv"] = mwebv
    curve_meta["gal_lat"] = b

    # set time step scale
    if time_scale == "trigger":
        # convert time to relative to discovery
        curve['relative_time'] = curve['TJD'] - curve_meta["TJD_discovery"]
        curve_meta["trigger"] = 0.0
    if time_scale == "first":
        # convert time to relative to 1st observation
        curve['relative_time'] = curve['TJD'] - curve['TJD'].iloc[0]
        curve_meta["trigger"] = convert_to_bin_timescale(curve_meta['TJD_discovery'] - curve['TJD'].iloc[0], bin_interval)
    if time_scale == "BTJD":
        curve['relative_time'] = curve['BTJD']
    if time_scale == "TJD":
        curve['relative_time'] = curve['TJD']

    if norm:
        curve[light], curve[uncert] = normalize(curve,uncert=uncert, light=light)

    # bin
    if to_bin:
        curve = bin_curves(curve, bin_interval, uncert=uncert)
    else:
        curve.index = curve['relative_time']


    return curve


def process_all_curves(parameters):
    """
    :return: iterator over all processed light_curves (PandaDataFrames) and their respective max light (TJD str)
    """
    light_curves = os.listdir("./TESS_data/ztf_data/") if parameters['ztf_tess'] else os.listdir("./TESS_data/light_curves_fausnaugh/")

    i = 0
    while i < len(light_curves):
        filename = light_curves[i]
        if parameters['ztf_tess']:
            light_curve, curve_meta = preprocess_ztf_tess(filename, parameters)
        else:
            light_curve, curve_meta = preprocess_tess(filename, parameters)

        if light_curve is not None:
            yield light_curve, curve_meta
        i += 1


def save_processed_curves(zip_name, params, fillval=0.0):
    mwebv_outliers = []
    directory = "./TESS_data/processed_curves/"
    for f in os.listdir(directory):
        os.remove(os.path.join(directory, f))

    for curve, meta in process_all_curves(params):
        # select cols
        if params['ztf_tess']:
            targets = ['tess_mag', 'tess_uncert', "r_mag", "r_uncert", 'g_mag', "g_uncert"]
        else:
            targets = ['cts', 'e_cts']
        df = curve[targets]
        # limit data points to those in range
        if 'range' in params:
            curve_range = params['range']
            df = df.loc[curve_range[0]:curve_range[1], :]
        df = df.fillna(fillval)
        tess_id = meta['IAU']
        print(tess_id)
        # filter out curves with mwebv above 0.5
        if meta['mwebv'] > 0.5:
            mwebv_outliers.append({"mwebv": meta['mwebv'], "class": meta['class'], 'gal_lat': meta['gal_lat'], 'IAU': tess_id})
        else:
            # add additional meta information
            for info in meta_targets:
                valid_timesteps = (df != fillval).any(axis=1)
                df[info] = pd.Series(meta[info], index=df[valid_timesteps].index)
                df.loc[~valid_timesteps, info] = fillval
            filename = f"lc_{tess_id}_{meta['ztf']}_processed.csv" if params['ztf_tess'] else f"lc_{tess_id}_processed.csv"
            # fill in missing timesteps
            if params['fill_values'] and 'range' in params:
                for t in range(curve_range[0], curve_range[1] + 1):
                    if t not in df.index:
                        df.loc[t] = np.zeros(shape=len(df.columns))
                        df = df.sort_index()
            df.to_csv(directory + filename)

    mwebv_df = pd.DataFrame(mwebv_outliers)
    mwebv_df.to_csv("./TESS_data/mwebv_outliers.csv", index=False)
    shutil.make_archive(zip_name, 'zip', directory)


if __name__ == "__main__":
    config = {
        "norm": True,
        "to_bin": True,
        "bin_interval": "0.5D",
        "time_scale": "trigger",
        "ztf_tess": True,
        "fill_values": True,
        "range": (-60, 60) # 30 divided by bin interval
    }
    # data, meta = preprocess_ztf_tess("2018gku_ZTF18abwoxal_detections.csv", config)
    # data_slice = data.loc[-60:60, :]
    # print(data_slice)

    # save all curves as CSVs
    save_processed_curves("processed_curves", config,  0.0)
