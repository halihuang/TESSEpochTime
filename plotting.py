from preprocess import preprocess_ztf_tess, preprocess_tess
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
        curve.plot.scatter(x=index, y=light, c=color, alpha=alpha, yerr=uncert, ylabel=y_label,
                                xlabel=xlabel, title=plot_title,
                                label=label, ax=ax)
    if max_line:
        t_max = find_max_light(curve, light, is_mag=is_mag)
        if t_max is not None:
            ax.axvline(t_max['relative_time'], color=max_color, linestyle="--", label=f"{label} Max")
    return ax


def display_all_passbands(df, meta, xlim, title_info="", ax=None, flip=True):
    if ax is None:
        ax = display_curve(df, meta, "tess_mag", "tess_uncert",color="gray", label="TESS", max_color="black")
    else:
        display_curve(df, meta, "tess_mag", "tess_uncert",color="gray", label="TESS", ax=ax, max_color="black")
    display_curve(df, meta, "r_mag", "r_uncert", color="orange", label="ZTF Red Passband", ax=ax, max_color="red")
    display_curve(df, meta, "g_mag", "g_uncert", color="green", label="ZTF Green Passband", ax=ax, max_line=False)
    if flip:
        ax.invert_yaxis()
    ax.set_xlim(xlim)
    ax.set_title(title_info)
    return ax


def ztf_plot(filename, params, xlim=[-30,70], save=False, save_dir="./TESS_data/ztf_tess_plots/all/", save_name="test"):
    fig, ax = plt.subplots(figsize=(12, 3))
    # plot normalized
    df, meta = preprocess_ztf_tess(filename, params)
    df = df.loc[xlim[0]:xlim[1], :]
    if df is not None and not df.empty:
        display_all_passbands(df, meta, flip=True, xlim=xlim, ax=ax)
        df.to_csv("./TESS_data/ztf_tess_plots/curves/no_norm/" + filename)

        fig.suptitle(format_title(meta), y=1.35)
        if save:
            fig.savefig(save_dir + save_name, bbox_inches="tight")
    if save:
        plt.close(fig)
    return ax


def tess_plot(tess_curve_name, params, save=False, save_dir="", xlim=[-30,30]):
    tess_file_name = f'lc_{tess_curve_name}_cleaned'
    params['to_bin'] = False
    fig, ax = plt.subplots()
    df, meta = preprocess_tess(tess_file_name, params)
    if df is not None:
        display_curve(df, meta, "cts", "e_cts", ax=ax, is_mag=False, alpha=0.2)
        params['to_bin'] = True

        df1, meta = preprocess_tess(tess_file_name, params)
        display_curve(df1, meta, "cts", "e_cts", ax=ax, is_mag=False, alpha=1.0, color="blue", max_color="blue", label="binned")

        ax.set_title(format_title(meta))
        ax.set_xlim(xlim)
        if save:
            fig.savefig(save_dir + tess_curve_name, bbox_inches="tight")
            plt.close(fig)
    return ax


