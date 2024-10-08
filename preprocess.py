import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import extinction
from astroquery.ipac.irsa.irsa_dust import IrsaDust
import astropy.coordinates as coord
import astropy.units as u
import shutil
from scipy.optimize import minimize_scalar
import glob


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
max_lights = pd.read_csv("./TESS_data/time_of_peak.csv", index_col="object_id")


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


def find_by_tess_name(tess_name):
    return glob.glob(f"./TESS_data/ztf_data/{tess_name}_*.csv")[0].split("\\")[1]


def preprocess_ztf(filename, parameters):
    # lightcurve data
    curve = pd.read_csv("./TESS_data/ztf_data/" + filename)
    split = filename.split("_")
    tess_curve_name = split[0]
    # convert from MJD to JD to TJD
    curve['TJD'] = curve['julian_date'] - 2457000.0
    # information about transient
    curve_meta = get_curve_meta(tess_curve_name)
    # return nothing if no info found
    if curve_meta is None:
        return None, None
    curve_meta['ztf'] = split[1]
    # split into passbands and create ZTF df
    ztf_data = {}
    passbands = {'r': "ZTF_r", 'g': "ZTF_g"}
    for pb, fid in passbands.items():
        pb_df = curve[curve['filter'] == fid]
        light_str = f'{pb}_flux' if not parameters['convert_to_mag'] else f'{pb}_mag'
        uncert_str = f'{pb}_uncert'
        if not pb_df.empty:
            pb_wavelength = 6215 if pb == "r" else 4767
            to_process_df = pd.DataFrame({"TJD": pb_df['TJD'], light_str: pb_df['flux'], uncert_str: pb_df["flux_unc"]})
            processed_pb = preprocess(to_process_df, curve_meta, light=light_str, uncert=uncert_str,
                                      to_bin=parameters['to_bin'], bin_interval=parameters['bin_interval'],
                                      time_scale=parameters['time_scale'], norm=parameters['norm'],
                                      median_filter=False, remove_extinction=parameters['remove_extinction'],
                                      convert_to_mag=parameters['convert_to_mag'], pb_wavelength=pb_wavelength)
            if not parameters['to_bin']:
                ztf_data[light_str] = remove_duplicate_indices(processed_pb[light_str])
                ztf_data[uncert_str] = remove_duplicate_indices(processed_pb[uncert_str])
            else:
                ztf_data[light_str] = processed_pb[light_str]
                ztf_data[uncert_str] = processed_pb[uncert_str]
        else:
            ztf_data[light_str] = pd.Series(dtype='float64')
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
                           median_filter= parameters['median_filter'], window_size= parameters['window_size'],
                           remove_extinction=parameters['remove_extinction'])
    light_col = "tess_mag" if parameters['convert_to_mag'] else "tess_flux"
    tess_df = pd.DataFrame({light_col: processed['cts'], "tess_uncert": processed['e_cts']})
    return tess_df, curve_meta


def preprocess_ztf_tess(ztf_filename, params):
    ztf_df, df_meta = preprocess_ztf(ztf_filename, params)
    if df_meta is None:
        return None, None
    tess_filename = f'lc_{df_meta["IAU_name"]}_cleaned'
    tess_df, _ = preprocess_tess(tess_filename, params)
    light_unit = "flux" if not params["convert_to_mag"] else "mag"

    exposure_time = 1425.6 if df_meta["sector"] in range(1, 27) else 475.2
    tess_df[f"tess_{light_unit}"] /= exposure_time
    tess_df["tess_uncert"] /= exposure_time
    curve = ztf_df.join(tess_df, how="outer")
    if curve.index.dtype != 'float64':
        curve.index = curve.index.astype('float64', copy=False)

    def get_correction_df(band_name, df):
        if not params['to_bin']:
            df["relative_time"] = df.index
            df = bin_curves(df, params["bin_interval"], uncert="tess_uncert")
        diff_df = df[~df[f"tess_{light_unit}"].isnull() & ~df[band_name].isnull()].loc[-5.0:, :]
        return diff_df

    def optimize_offset(band_name, df):
        rescale_df = get_correction_df(band_name, df)
        if not rescale_df.empty:
            ztf = rescale_df[band_name]
            tess = rescale_df[f"tess_{light_unit}"]

            def diff(m):
                return np.mean(np.square(m * tess - ztf))

            scale_factor = minimize_scalar(diff).x
            if scale_factor >= 0.025:
                df[f"tess_{light_unit}"] *= scale_factor
                df[f"tess_uncert"] *= scale_factor
                return True
        return False

    def diff_correction(band_name, df):
        diff_df = get_correction_df(band_name, df)
        if not diff_df.empty and len(diff_df.index) >= 2:
            correction = (diff_df[band_name] - diff_df[f"tess_{light_unit}"]).mean()
            df[f"tess_{light_unit}"] += correction
            return True
        return False

    rescaled = False
    if params["optimize_scale"]:
        rescaled = optimize_offset(f"r_{light_unit}", curve)
        if not rescaled:
            rescaled = optimize_offset(f"r_{light_unit}", curve)

    if not rescaled:
        scale = params["scale_factor"]
        curve[f"tess_{light_unit}"] *= scale
        curve[f"tess_uncert"] *= scale

    corrected = diff_correction(f"r_{light_unit}", curve)
    if not corrected:
        corrected = diff_correction(f"g_{light_unit}", curve)

    curve[f"tess_{light_unit}"] += params["manual_diff_corr"]
    # print(rescaled, corrected)
    return curve, df_meta


def create_params_obj(sub_bg_model=False, remove_extinction=True, median_filter=True, window_size="1.5D",
                      to_bin=True, norm=True, bin_interval="0.5D", time_scale="trigger",
                      convert_to_mag=False):
    return {
        "norm": norm,
        "to_bin": to_bin,
        "bin_interval": bin_interval,
        "time_scale": time_scale,
        "convert_to_mag": convert_to_mag,
        "sub_bg_model": sub_bg_model,
        "median_filter": median_filter,
        "window_size": window_size,
        "remove_extinction": remove_extinction,
        # ztf_tess alignment params
        "scale_factor": 180,
        "optimize_scale": True,
        "manual_diff_corr": 0
    }


def preprocess(curve, curve_meta, light="cts", uncert="e_cts", sub_bg_model=False, remove_extinction=True,
               to_bin=True, norm=True, bin_interval="0.5D", time_scale="trigger", convert_to_mag=False,
               median_filter=True, window_size="1.5D", pb_wavelength=7865):
    """
    processes data of single lightcurve


    :param curve: panda dataframe containing light curve data
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
    :param median_filter: if median window-filtering should be applied
    :param window_size: size of window for median window filtering
    :param pb_wavelength: central passband wavelength in Angstroms
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

    # median window filtering
    if median_filter:
        outliers = []
        filter_df = curve[[light, "relative_time"]].copy()
        filter_df.index = pd.TimedeltaIndex(curve["relative_time"], unit="D")

        def median_window_filtering(window):
            median = window[light].median()
            abl_deviations = (window[light] - median).abs()
            mad = abl_deviations.median()
            window_outliers = window[abl_deviations >= 3 * mad]
            if not window_outliers.empty:
                outliers.extend(window_outliers["relative_time"].tolist())
            return 0

        filter_df.resample(window_size, origin="start").apply(median_window_filtering)
        curve = curve[~curve["relative_time"].isin(outliers)]

    if convert_to_mag:
        curve[light] -= curve[light].min() - 1.0
        if to_bin:
            curve = bin_curves(curve, bin_interval, uncert=uncert)
            curve_meta['interval'] = bin_interval
        # convert
        sector = curve_meta['sector']
        curve[uncert] = convert_ects_to_mag(curve[light], curve[uncert])
        curve[light] = convert_cts_to_mag(curve[light], sector)
        # sigma clipping by e_cts
        curve = sigma_clip(curve, uncert)
    # offset so whole curve is positive
    elif light == "cts":
        pretrigger = curve[curve["relative_time"] < -5.0][light]
        if not pretrigger.empty and not pretrigger.isnull().all():
            curve[light] -= pretrigger.median()

    if remove_extinction:
        ra = curve_meta["ra"]
        dec = curve_meta["dec"]
        flux_in = curve[light]
        bandpass_wavelengths = np.array([pb_wavelength,])

        # Get Milky Way E(B-V) Extinction
        coo = coord.SkyCoord(ra, dec, frame='icrs', unit=(u.hourangle, u.deg))
        b = coo.galactic.b.value
        dust = IrsaDust.get_query_table(coo, section='ebv')
        mwebv = dust['ext SandF mean'][0]

        # Remove extinction from light curves
        # (Using negative a_v so that extinction.apply works in reverse and removes the extinction)
        extinction_per_passband = extinction.fitzpatrick99(wave=bandpass_wavelengths, a_v=-3.1 * mwebv, r_v=3.1, unit='aa')
        if convert_to_mag:
            flux_out = flux_in + extinction_per_passband[0]
        else:
            flux_out = extinction.apply(extinction_per_passband[0], flux_in, inplace=False)

        curve[light] = flux_out
        curve_meta["mwebv"] = mwebv
        curve_meta["gal_lat"] = b

    if norm:
        curve[light], curve[uncert] = normalize(curve, uncert=uncert, light=light)

    # bin
    if not convert_to_mag and to_bin:
        curve = bin_curves(curve, bin_interval, uncert=uncert)

    if not to_bin:
        curve.index = curve['relative_time']
        curve_meta['interval'] = bin_interval

    return curve


def get_max_light(tess_name):
    max_light = max_lights.loc[tess_name, "median_time_of_max"]
    uncert = max_lights.loc[tess_name, "uncertainty"]
    return max_light, uncert


def save_processed_curves(params, to_process_lst=[], ztf_tess=True, enable_final_processing=True, meta_targets=[], fill_missing=True,
                          mask_val=0.0, curve_range=None, reset=False, directory="./TESS_data/processed_curves/", curve_labels=None):
    mwebv_outliers = []
    if reset:
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
            passbands = ["tess", "r", "g"] if ztf_tess else ["tess"]
            light_unit = "mag" if params["convert_to_mag"] else "flux"
            targets = [f"{pb}_{light_unit}" for pb in passbands] + [f"{pb}_uncert" for pb in passbands]
            df = curve[targets]
            tess_id = meta['IAU_name']
            print(tess_id)
            # limit data points to those in range
            if curve_range is not None:
                df = df.loc[curve_range[0]:curve_range[1], :]

            # filter out curves with mwebv above 0.5
            if meta['mwebv'] > 0.6:
                mwebv_outliers.append({"mwebv": meta['mwebv'], "class": meta['classification'], 'gal_lat': meta['gal_lat'], 'IAU_name': tess_id})
            else:
                if enable_final_processing:
                    # decide if passband gets allowed based on label
                    if curve_labels is not None and ztf_tess:
                        curve_label = curve_labels.set_index("curve_name").loc[tess_id, :]
                        include_tess = curve_label[["tess_maybe", "tess_good", "tess_great"]].any()
                        include_ztf = curve_label[["ztf_maybe", "ztf_good", "ztf_great"]].any()
                        if not include_tess:
                            df[[f"tess_{light_unit}", "tess_uncert"]] = np.NaN
                        if not include_ztf:
                            df[[f"r_{light_unit}", f"g_{light_unit}", "g_uncert", "r_uncert"]] = np.NaN

                    # add additional meta information
                    for info in meta_targets:
                        valid_timesteps = (df != mask_val).any(axis=1)
                        df[info] = pd.Series(meta[info], index=df[valid_timesteps].index)
                        df.loc[~valid_timesteps, info] = mask_val
                    # fill in missing timesteps
                    if fill_missing:
                        df = df.fillna(mask_val)
                        interval_val = float(params["bin_interval"].split("D")[0])
                        if curve_range is None:
                            raise Exception("Needs range to be set")
                        for t in np.arange(curve_range[0], curve_range[1] + interval_val, interval_val):
                            if t not in df.index:
                                df.loc[t] = np.full(shape=len(df.columns), fill_value=mask_val)
                                df = df.sort_index()

                    max_light, uncert = get_max_light(tess_id)
                    df["max_light"] = max_light
                    df["max_uncert"] = uncert
                filename = f"lc_{tess_id}_{meta['ztf']}_processed.csv" if ztf_tess else f"lc_{tess_id}_processed.csv"
                df.to_csv(directory + filename)

    mwebv_df = pd.DataFrame(mwebv_outliers)
    mwebv_df.to_csv("./TESS_data/mwebv_outliers.csv", index=False)

if __name__ == "__main__":
    config = {
        "norm": False,
        "to_bin": False,
        "bin_interval": "0.5D",
        "time_scale": "trigger",
        'convert_to_mag': False,
        'sub_bg_model': False,
        'remove_extinction': True,
        "median_filter": True,
        "window_size": "2.5D",
        # ztf_tess alignment params
        "scale_factor": 180,
        "optimize_scale": True,
        "manual_diff_corr": 0
    }
    zip_name = "processed_curves_ztf_tess_unbinned_good_great"
    curve_lim = (-30, 70)
    fill = False
    maskval = 0.0
    meta_targets = ["mwebv"]
    ztf_tess = True
    label_formatting = True

    # save all curves as CSVs for training
    labels = pd.read_csv("./TESS_data/curve_labels.csv")  # set to None to ignore labels
    tess_good = (labels["tess_good"] == True)
    tess_maybe = (labels["tess_maybe"] == True)
    tess_great = (labels["tess_great"] == True)
    ztf_good = (labels["ztf_good"] == True)
    ztf_maybe = (labels["ztf_maybe"] == True)
    ztf_great = (labels["ztf_great"] == True)
    bad_scale = (labels["bad_scale"] == True)

    # config this to adjust which part of dataset
    desired_labels = labels[~bad_scale & ((ztf_maybe & (tess_maybe | tess_good | tess_great)) | ztf_good | ztf_great)]["curve_name"]
    to_process = [find_by_tess_name(label) for label in desired_labels] # set to None to process all

    # # # process all
    # to_process = None
    # labels = None
    save_processed_curves(config, to_process_lst=to_process, reset=True,
                          meta_targets=meta_targets, mask_val=maskval, fill_missing=fill, curve_range=curve_lim,
                          ztf_tess=ztf_tess, curve_labels=labels, enable_final_processing=label_formatting)
    if ztf_tess:
        # 12 curves that do worse with optimized scaling
        config["optimize_scale"] = False
        to_process = ["2018kfv_ZTF18acwutbr_exitcode57.csv", "2018koy_ZTF18adaifep_exitcode0.csv",
                      "2019bwu_ZTF19aamsjlt_exitcode61.csv", "2020dya_ZTF20aasijew_exitcode56.csv",
                      "2020ebr_ZTF20aarjgox_exitcode56.csv", "2021abbl_ZTF21achcwnd_exitcode56.csv",
                      "2021dsb_ZTF21aamucom_exitcode56.csv", "2021hup_ZTF21aarhnwn_exitcode0.csv",
                      "2021rgm_ZTF21abilrxd_exitcode61.csv","2021ucq_ZTF21abouexm_exitcode56.csv",
                      "2021xzf_ZTF21abyfxqr_exitcode56.csv", "2022dma_ZTF22aabwfss_exitcode0.csv"]
        save_processed_curves(config, to_process_lst=to_process,
                              meta_targets=meta_targets, mask_val=maskval, fill_missing=fill, curve_range=curve_lim,
                              ztf_tess=ztf_tess, curve_labels=labels, enable_final_processing=label_formatting)

        # 2018ksr
        config["scale_factor"] = 80
        save_processed_curves(config, to_process_lst=["2018ksr_ZTF18adaivyd_exitcode56.csv"],
                              meta_targets=meta_targets, mask_val=maskval, fill_missing=fill, curve_range=curve_lim,
                              ztf_tess=ztf_tess, curve_labels=labels, enable_final_processing=label_formatting)

        # 2020fcw, 2022eat
        config["scale_factor"] = 180
        config["manual_diff_corr"] = 5000
        to_process = ["2020fcw_ZTF20aattotq_exitcode0.csv", "2022eat_ZTF22aacrugz_exitcode57.csv"]
        save_processed_curves(config, to_process_lst=to_process,
                              meta_targets=meta_targets, mask_val=maskval, fill_missing=fill, curve_range=curve_lim,
                              ztf_tess=ztf_tess, curve_labels=labels, enable_final_processing=label_formatting)

        # 2018hxq
        config["manual_diff_corr"] = 600
        save_processed_curves(config, to_process_lst=["2018hxq_ZTF18abyiusv_exitcode62.csv"],
                              meta_targets=meta_targets, mask_val=maskval, fill_missing=fill, curve_range=curve_lim,
                              ztf_tess=ztf_tess, curve_labels=labels, enable_final_processing=label_formatting)

        # 2019axj, 2022dyu, 2022eaz, 2022een
        config["manual_diff_corr"] = 1200
        to_process = ["2019axj_ZTF19aajwjwq_exitcode61.csv", "2022een_ZTF22aacrwal_exitcode0.csv",
                      "2022dyu_ZTF22aacdkzo_exitcode56.csv","2022eaz_ZTF18aahvpcy_exitcode0.csv"]
        save_processed_curves(config, to_process_lst=to_process,
                              meta_targets=meta_targets, mask_val=maskval, fill_missing=fill, curve_range=curve_lim,
                              ztf_tess=ztf_tess, curve_labels=labels, enable_final_processing=label_formatting)

        # 2019bip
        config["scale_factor"] = 800
        config["manual_diff_corr"] = 6500
        save_processed_curves(config, to_process_lst=["2019bip_ZTF19aallimd_exitcode56.csv"],
                              meta_targets=meta_targets, mask_val=maskval, fill_missing=fill, curve_range=curve_lim,
                              ztf_tess=ztf_tess, curve_labels=labels, enable_final_processing=label_formatting)

        # 2018lot
        config["scale_factor"] = 50
        config["manual_diff_corr"] = -10000
        save_processed_curves(config, to_process_lst=["2018lot_ZTF17aaabgiw_exitcode0.csv"],
                              meta_targets=meta_targets, mask_val=maskval, fill_missing=fill, curve_range=curve_lim,
                              ztf_tess=ztf_tess, curve_labels=labels, enable_final_processing=label_formatting)

    shutil.make_archive(f'./TESS_data/processed_zips/{zip_name}', 'zip', "./TESS_data/processed_curves/")
    print(f"Saved to:{zip_name}")