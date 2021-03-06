import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import extinction
from astroquery.ipac.irsa.irsa_dust import IrsaDust
import astropy.coordinates as coord
import astropy.units as u
import shutil


# transients = pd.read_csv("./TESS_data/AT_count_transients_s1-47.txt",
#                          names=["sector", "ra", "dec", "discovery_mag", "discovery_date", "type", "classification", "IAU_name",
#                                 "discovery_survey","cam", "ccd", "col", "row"], delim_whitespace=True)
# transients.index = transients["IAU_name"]


def set_earliest_time(row):
    discovery_times = row["discovery_date"].split("/")
    row["ra"] = row["ra"].split("/")[1]
    row["dec"] = row["dec"].split("/")[1]
    ts_series = pd.Series([pd.Timestamp(time) for time in discovery_times])
    i = ts_series.idxmin()
    earliest_disc_time = ts_series[i]
    row["discovery_survey"] = row["discovery_survey"].split("/")[i]
    # set TJD disc time to spot
    row["discovery_date"] = earliest_disc_time.to_julian_date() - 2457000.0
    return row


transients = pd.read_csv("./TESS_data/tns_info2.csv")
transients.index = transients["IAU_name"]
transients = transients.apply(set_earliest_time, axis=1)


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
    binned.index = binned.index / pd.to_timedelta("1D")
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


def convert_cts_to_mag(cts, sec):
    exposure_time = 1425.6 if sec in range(1, 27) else 475.2
    zero_point = 20.44
    mag = zero_point - 2.5 * np.log10(cts / exposure_time)
    return mag


def convert_ects_to_mag(cts, e_cts):
    delta_mag = (2.5 / np.log(10)) * (e_cts / cts)
    return delta_mag


def remove_duplicate_indices(data_col):
    duplicates = data_col.index.duplicated()
    return data_col.loc[~duplicates]


def get_curve_meta(curve_name):
    curve_meta = transients.loc[curve_name, :].copy()
    if curve_meta.empty:
        return None
    else:
        return curve_meta


def find_max_light(curve, light_col, is_mag=False):
    if curve[light_col].isnull().all():
        return None
    id_max = curve[light_col].idxmin() if is_mag else curve[light_col].idxmax()
    max_data = curve.loc[id_max, :].copy()
    max_data['relative_time'] = id_max
    return max_data


def format_title(curve_meta, title_info=""):
    title = f"{title_info}\n"\
        f"{curve_meta['IAU_name']}\n Class: {curve_meta['classification']}, Sector: {curve_meta['sector']} " \
        f"\nCoords:{curve_meta['ra'], curve_meta['dec']}, \nDiscovery TJD: {curve_meta['discovery_date']}, " \
        f"Survey: {curve_meta['discovery_survey']}"
    if 'mwebv' in curve_meta:
        title += f"\nmwebv: {curve_meta['mwebv']}"
    return title


def display_curve(light_curve, curve_meta, light, uncert,  index="index", color="00000", label="curve", plot_title="",
                  xlabel="Days since trigger", alpha=0.5, ax=None, max_line=True, max_color="red", is_mag=True):

    if light_curve[light].empty or light_curve[light].isnull().all():
        return

    if not plot_title:
        plot_title = format_title(curve_meta)
    curve = light_curve.copy()
    curve['index'] = light_curve.index
    y_label = "mag" if is_mag else "cts"
    if ax is None:
        ax = curve.plot.scatter(x=index, y=light, c=color, alpha=alpha, yerr=uncert, ylabel=y_label,
                                xlabel=xlabel, title=plot_title,
                                label=label)
    else:
        ax = curve.plot.scatter(x=index, y=light, c=color, alpha=alpha, yerr=uncert, ylabel=y_label,
                                xlabel=xlabel, title=plot_title,
                                label=label, ax=ax)
    if max_line:
        t_max = find_max_light(curve, light, is_mag=is_mag)
        if t_max is not None:
            ax.axvline(t_max['relative_time'], color=max_color, linestyle="--")
    return ax


def display_all_passbands(df, meta, xlim, title_info="", ax=None, flip=True):
    if ax is None:
        ax = display_curve(df, meta, "tess_mag", "tess_uncert",color="gray", label="TESS", max_color="black")
    else:
        ax = display_curve(df, meta, "tess_mag", "tess_uncert",color="gray", label="TESS", ax=ax, max_color="black")
    display_curve(df, meta, "r_mag", "r_uncert", color="orange", label="ZTF Red Passband", ax=ax, max_color="red")
    display_curve(df, meta, "g_mag", "g_uncert", color="green", label="ZTF Green Passband", ax=ax, max_color="green",
                  xlabel=f"units ({meta['interval']}) relative to discovery")
    if flip:
        ax.invert_yaxis()
    ax.set_xlim(xlim)
    ax.set_title(title_info)
    return ax


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
                                      time_scale=parameters['time_scale'], norm=parameters['norm'],
                                      remove_extinction=parameters['remove_extinction'], is_mag=True, convert_to_mag=False)
            if not parameters['to_bin']:
                ztf_data[mag_str] = remove_duplicate_indices(processed_pb[mag_str])
                ztf_data[uncert_str] = remove_duplicate_indices(processed_pb[uncert_str])
            else:
                ztf_data[mag_str] = processed_pb[mag_str]
                ztf_data[uncert_str] = processed_pb[uncert_str]
        else:
            ztf_data[mag_str] = pd.Series(dtype='float64')
            ztf_data[uncert_str] = pd.Series(dtype='float64')

    ztf_df = pd.DataFrame(ztf_data)
    ztf_df.index = ztf_df.index.rename("relative_time")
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
                           norm=parameters['norm'], sub_bg_model=parameters['sub_bg_model'],
                           remove_extinction=parameters['remove_extinction'], is_mag=False)
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
    tess_filename = f'lc_{df_meta["IAU_name"]}_cleaned'
    params['convert_to_mag'] = True
    tess_df, _ = preprocess_tess(tess_filename, params)
    output = ztf_df.join(tess_df, how="outer")
    if output.index.dtype != 'float64':
        output.index = output.index.astype('float64', copy=False)

    diff_df = output[~output["tess_mag"].isnull() & ~output["r_mag"].isnull()]
    if not diff_df.empty:
        output["tess_mag"] += (diff_df["r_mag"] - diff_df["tess_mag"]).mean()
    return output, df_meta


def preprocess(curve, curve_meta, light="cts", uncert="e_cts", sub_bg_model=False, remove_extinction=True,
               convert_to_mag=False, to_bin=True, norm=True, bin_interval="0.5D", time_scale="trigger", is_mag=False):
    """
    processes data of single lightcurv

    :param curve: panda dataframe containint light curve data
    :param curve_meta: meta information obtained from AT_transients
    :param light: str key for light column (cts or mag) of DataFrame
    :param uncert: str key for uncert column of Dataframe
    :param sub_bg_model: if background model flux should be subtracted from real
    :param remove_extinction:
    :param bin_interval: Day: D, Minute: T, Second: S
    :param norm: normalizes curve
    :param to_bin: bins data according to bin interval
    :param convert_to_mag: converts cts to magnitude
    :param time_scale: one of ["first", "trigger", "BTJD", "TJD"], determines whether to index relative to first
    observation or trigger, BTJD, or TJD time"
    :param is_mag:

    :return: processed light curve in PandaDataframe
    """

    # set time step scale
    if time_scale == "trigger":
        # convert time to relative to discovery
        curve['relative_time'] = curve['TJD'] - curve_meta["discovery_date"]
        curve_meta["trigger"] = 0.0
    if time_scale == "first":
        # convert time to relative to 1st observation
        curve['relative_time'] = curve['TJD'] - curve['TJD'].iloc[0]
        curve_meta["trigger"] = convert_to_bin_timescale(curve_meta['discovery_date'] - curve['TJD'].iloc[0],
                                                         bin_interval)

    # sub bg flux
    if sub_bg_model:
        if not curve['bkg_model'].isnull().all():
            curve[light] = curve[light] - curve['bkg_model']

    # sigma clipping by e_cts
    curve = sigma_clip(curve, uncert)

    # sigma clipping by cts
    curve = sigma_clip(curve, light)

    if convert_to_mag:
        curve[light] -= curve[light].min() - 1.0

    # bin
    if (convert_to_mag or is_mag) and to_bin:
        curve = bin_curves(curve, bin_interval, uncert=uncert)
        curve_meta['interval'] = bin_interval

    # print(curve)
    # print(curve[curve['relative_time'] == -20.81408999999985])

    # convert to magnitude
    if convert_to_mag:
        sector = curve_meta['sector']
        curve[uncert] = convert_ects_to_mag(curve[light], curve[uncert])
        curve[light] = convert_cts_to_mag(curve[light], sector)
        is_mag = True

    # sigma clipping by e_cts
    curve = sigma_clip(curve, uncert)

    if remove_extinction:
        ra = curve_meta["ra"]
        dec = curve_meta["dec"]
        flux_in = curve[light]
        bandpass_wavelengths = np.array([7865,])

        # Get Milky Way E(B-V) Extinction
        coo = coord.SkyCoord(ra, dec, frame='icrs', unit=(u.hourangle, u.deg))
        b = coo.galactic.b.value
        dust = IrsaDust.get_query_table(coo, section='ebv')
        mwebv = dust['ext SandF mean'][0]

        # Remove extinction from light curves
        # (Using negative a_v so that extinction.apply works in reverse and removes the extinction)
        extinction_per_passband = extinction.fitzpatrick99(wave=bandpass_wavelengths, a_v=-3.1 * mwebv, r_v=3.1, unit='aa')
        if is_mag:
            flux_out = flux_in + extinction_per_passband[0]
        else:
            flux_out = extinction.apply(extinction_per_passband[0], flux_in, inplace=False)

        curve[light] = flux_out
        curve_meta["mwebv"] = mwebv
        curve_meta["gal_lat"] = b

    if norm:
        curve[light], curve[uncert] = normalize(curve, uncert=uncert, light=light)

    # bin
    if not is_mag and to_bin:
        curve = bin_curves(curve, bin_interval, uncert=uncert)

    if not to_bin:
        curve.index = curve['relative_time']
        curve_meta['interval'] = bin_interval

    return curve


def save_processed_curves(zip_name, params, to_process_lst=[], ztf_tess=True, enable_final_processing=True, meta_targets=[], fill_missing_timesteps=True,
                          mask_val=-1.0, curve_range=None):
    mwebv_outliers = []
    directory = "./TESS_data/processed_curves/"
    for f in os.listdir(directory):
        os.remove(os.path.join(directory, f))

    if not to_process_lst:
        to_process_lst = os.listdir("./TESS_data/ztf_data/") if ztf_tess else \
            os.listdir("./TESS_data/light_curves_fausnaugh/")

    for filename in to_process_lst:
        if ztf_tess:
            curve, meta = preprocess_ztf_tess(filename, params)
        else:
            curve, meta = preprocess_tess(filename, params)

        if curve is not None and not curve.empty:
            if ztf_tess:
                targets = ['tess_mag', 'tess_uncert', "r_mag", "r_uncert", 'g_mag', "g_uncert"]
            else:
                targets = ['cts', 'e_cts']
            df = curve[targets]
            tess_id = meta['IAU_name']
            print(tess_id)
            # limit data points to those in range
            if curve_range is not None:
                df = df.loc[curve_range[0]:curve_range[1], :]

            # filter out curves with mwebv above 0.5
            if meta['mwebv'] > 0.5:
                mwebv_outliers.append({"mwebv": meta['mwebv'], "class": meta['class'], 'gal_lat': meta['gal_lat'], 'IAU_name': tess_id})
            else:
                if enable_final_processing:
                    df = df.fillna(mask_val)
                    # add additional meta information
                    for info in meta_targets:
                        valid_timesteps = (df != mask_val).any(axis=1)
                        df[info] = pd.Series(meta[info], index=df[valid_timesteps].index)
                        df.loc[~valid_timesteps, info] = mask_val
                    # fill in missing timesteps
                    if fill_missing_timesteps:
                        if curve_range is None:
                            raise Exception("Needs range to be set")
                        for t in range(curve_range[0], curve_range[1] + 1):
                            if t not in df.index:
                                df.loc[t] = np.zeros(shape=len(df.columns))
                                df = df.sort_index()
                filename = f"lc_{tess_id}_{meta['ztf']}_processed.csv" if ztf_tess else f"lc_{tess_id}_processed.csv"
                df.to_csv(directory + filename)

    mwebv_df = pd.DataFrame(mwebv_outliers)
    mwebv_df.to_csv("./TESS_data/mwebv_outliers.csv", index=False)
    shutil.make_archive(f'./TESS_data/processed_zips/{zip_name}', 'zip', directory)


if __name__ == "__main__":
    config = {
        "norm": True,
        "to_bin": True,
        "bin_interval": "0.5D",
        "time_scale": "trigger",
        'convert_to_mag': True,
        'sub_bg_model': False,
        'remove_extinction': True
    }
    # data, meta = preprocess_ztf_tess("2020ir_ZTF20aabpmdj_detections.csv", config)
    # data, meta = preprocess_tess("lc_2021hi_cleaned", config)
    # save all curves as CSVs for training
    import warnings
    import glob

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labels = pd.read_csv("./TESS_data/curve_labels.csv")
        # config this to adjust which dataset
        good = (labels["tess_good"] == True)
        maybe = (labels["tess_maybe"] == True)
        great = (labels["tess_great"] == True)
        desired_labels = labels[good]["curve_name"]
        # ztf_tess
        to_process = [glob.glob(f"./TESS_data/ztf_data/{label}_*.csv")[0].split("\\")[1] for label in desired_labels]
        save_processed_curves("processed_curves", config, to_process_lst=to_process,
                              meta_targets=["mwebv", "discovery_mag"], mask_val=-1.0,
                              fill_missing_timesteps=True, curve_range=(-60, 60), ztf_tess=True)

