from preprocess import preprocess_ztf_tess, preprocess_tess, get_curve_meta
import matplotlib.pyplot as plt


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


def display_curve(light_curve, curve_meta, light, uncert,  index="index", color="00000", label="curve", plot_title=None,
                  xlabel="Days since trigger", alpha=0.5, ax=None, max_line=True, max_color="red", is_mag=True):

    if light_curve[light].empty or light_curve[light].isnull().all():
        return ax

    if plot_title is None:
        plot_title = format_title(curve_meta)
    curve = light_curve.copy()
    curve['index'] = light_curve.index
    y_label = "mag" if is_mag else "flux"
    if ax is None:
        ax = curve.plot.scatter(x=index, y=light, c=color, alpha=alpha, yerr=uncert, ylabel=y_label,
                                xlabel=xlabel, title=plot_title,
                                label=label)
    else:
        curve.plot.scatter(x=index, y=light, c=color, alpha=alpha, yerr=uncert, ylabel=y_label,
                                xlabel=xlabel, title=plot_title,
                                label=label, ax=ax)
    if max_line:
        t_max = find_max_light(curve, light, is_mag=is_mag)
        if t_max is not None:
            ax.axvline(t_max['relative_time'], color=max_color, linestyle="--", label=f"{label} Max: {t_max['relative_time']}")
    ax.legend(fontsize=7, loc="upper right")
    return ax


def display_all_passbands(df, meta, xlim, title_info="", ax=None, is_mag=False):
    light_unit = "mag" if is_mag else "flux"
    if ax is None:
        ax = display_curve(df, meta, f"tess_{light_unit}", "tess_uncert",color="gray", label="TESS", max_color="black", is_mag=is_mag)
    else:
        display_curve(df, meta, f"tess_{light_unit}", "tess_uncert",color="gray", label="TESS", ax=ax, max_color="black", is_mag=is_mag)
    display_curve(df, meta, f"r_{light_unit}", "r_uncert", color="orange", label="ZTF Red Passband", ax=ax, max_color="red", is_mag=is_mag)
    display_curve(df, meta, f"g_{light_unit}", "g_uncert", color="green", label="ZTF Green Passband", ax=ax, max_line=False, is_mag=is_mag)
    if is_mag:
        ax.invert_yaxis()
    ax.set_xlim(xlim)
    ax.set_title(title_info)
    return ax


def ztf_tess_plot(filename, params, xlim=[-30,70], save=False, save_dir="./TESS_data/ztf_tess_plots/all/", save_name="test"):
    fig, ax = plt.subplots(figsize=(12, 3))
    df, meta = preprocess_ztf_tess(filename, params)
    df = df.loc[xlim[0]:xlim[1], :]
    if df is not None and not df.empty:
        display_all_passbands(df, meta, is_mag=params["convert_to_mag"], xlim=xlim, ax=ax)
        df.to_csv("./TESS_data/ztf_tess_plots/curves/no_norm/" + filename)

        fig.suptitle(format_title(meta), y=1.35)
        if save:
            fig.savefig(save_dir + save_name, bbox_inches="tight")
            plt.close(fig)
    return df, ax


def tess_plot(tess_curve_name, parameters, unbinned=True, save=False, save_dir="", xlim=[-30, 30], ax=None):
    tess_file_name = f'lc_{tess_curve_name}_cleaned'
    params = parameters.copy()
    params['to_bin'] = True
    if ax is None:
        fig, ax = plt.subplots()
    df, meta = preprocess_tess(tess_file_name, params)
    light_col = "tess_mag" if params["convert_to_mag"] else "tess_flux"
    if df is not None:
        if unbinned:
            params['to_bin'] = False
            df1, meta = preprocess_tess(tess_file_name, params)
            display_curve(df1, meta, light_col, "tess_uncert", ax=ax, is_mag=False, alpha=0.2, color="blue", max_color="blue", label="unbinned")

        display_curve(df, meta, light_col, "tess_uncert", ax=ax, is_mag=False, alpha=0.8)

        ax.set_title(format_title(meta))
        ax.set_xlim(xlim)
        if save:
            fig.savefig(save_dir + tess_curve_name, bbox_inches="tight")
            plt.close(fig)
    return df, ax


def plot_pb_seperated(filename, params, xlim=[-30,70], save=False, save_dir="./TESS_data/ztf_tess_plots/all/", save_name="test"):
    light_unit = "flux" if not params["convert_to_mag"] else "mag"
    tess_name = filename.split("_")[0]
    fig, ax = plt.subplots(3,1, figsize=(12, 10))
    df, meta = preprocess_ztf_tess(filename, params)
    if df is not None and not df.empty:
        df.to_csv("./TESS_data/ztf_tess_plots/curves/" + filename)
        fig.suptitle(format_title(meta), y=1.05)
        display_all_passbands(df, meta, is_mag=params["convert_to_mag"], xlim=xlim, ax=ax[0])
        display_curve(df, meta, f"r_{light_unit}", "r_uncert", color="orange", label="ZTF Red Passband",
                           max_color="red", is_mag=params["convert_to_mag"], ax=ax[1], plot_title="")
        display_curve(df, meta, f"g_{light_unit}", "g_uncert", color="green", label="ZTF Green Passband",
                           max_line=False, is_mag=params["convert_to_mag"], ax=ax[1], plot_title="")
        ax[1].set_xlim(xlim)
        tess_plot(tess_name, params, xlim=xlim, ax=ax[2])
        ax[2].set_title("")
        ax[2].set_ylabel("counts per sec")

        if save:
            fig.savefig(save_dir + save_name, bbox_inches="tight")
            plt.close(fig)
    return ax

