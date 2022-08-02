import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotting import display_all_passbands, format_title
from model import build_model
from model_config import params
from preprocess import get_curve_meta


def find_label_max_light(curve):
    tess_mag = curve[curve["tess_mag"] != params.maskval]['tess_mag']
    tess_t_max = tess_mag.idxmin()
    tess_max = curve.loc[tess_t_max, :]
    last_tess_datapoint = tess_mag.iloc[-1]

    ztf_r = curve[curve["r_mag"] != params.maskval]["r_mag"]
    if not ztf_r.empty:
        ztf_t_max = curve[curve["r_mag"] != params.maskval]["r_mag"].idxmin()
        ztf_max = curve.loc[ztf_t_max, :]

        if ztf_t_max > last_tess_datapoint and ztf_max["r_mag"] > tess_max["tess_mag"]:
            ztf_max["relative_time"] = ztf_t_max
            return ztf_max

    tess_max['relative_time'] = tess_t_max
    return tess_max


def format_data(data):
    data_dir = "./TESS_data/processed_curves/"
    x = np.zeros(shape=(len(data), params.timesteps, params.n_features))
    y = np.zeros(shape=(len(data), len(params.targets)))

    for i, csv in enumerate(data):
        df = pd.read_csv(data_dir + csv, usecols=params.features + ['relative_time'])
        df.index = df["relative_time"]
        df = df.fillna(params.maskval)
        passbands = {"tess": 1, "r": 2, "g": 3}
        combined_df = pd.DataFrame()
        for pb, id in passbands.items():
            pb_df = df[["relative_time", "mwebv"]].copy()
            pb_df["uncert"] = df[f"{pb}_uncert"]
            pb_df["mag"] = df[f"{pb}_mag"]
            pb_df["id"] = id
            pb_df[(pb_df["mag"] == params.maskval) & (pb_df["uncert"] == params.maskval)] = params.maskval

            for t in np.arange(params.curve_range[0], params.curve_range[1] + params.interval_val, params.interval_val):
                if t not in pb_df["relative_time"]:
                    pb_df.loc[t] = np.full(shape=len(pb_df.columns), fill_value=params.maskval)
                    pb_df = pb_df.sort_index()
            combined_df = pd.concat([pb_df, combined_df])

        max_light = find_label_max_light(df)
        x[i] = combined_df.sort_index().to_numpy()
        y[i] = max_light[params.targets].to_numpy()
    return x, y


def predict(model, test, iterations=100):
    test_len = test.shape[0]
    avg_t = np.zeros((test_len,1))
    avg_uncert = np.zeros((test_len,1))
    for i in range(iterations):
        distrib = model(test)
        max_t = distrib.mean().numpy()
        uncert = distrib.stddev().numpy()
        avg_t += max_t
        avg_uncert += uncert
    avg_t = avg_t.reshape(avg_t.shape[0])
    avg_uncert = avg_uncert.reshape(avg_uncert.shape[0])
    avg_t /= iterations
    avg_uncert /= iterations
    return avg_t, avg_uncert


def predict_real_time(real, model, test, iterations=100):
    uncert_axis = []
    mse_axis = []
    for i in range(1, params.timesteps+1):
        input_i = test.copy()
        input_i[:, i:, :] = 0.0
        max_t, uncert = predict(model, input_i, iterations)
        uncert_axis.append(uncert.mean())
        squared_error = np.square(real-max_t)
        mse_axis.append(squared_error.mean())
    return pd.DataFrame({"timesteps": range(1, params.timesteps+1), "mse": mse_axis, "avg_uncert":uncert_axis})


def load_model(model_name):
    model_path = f"./TESS_data/models/{model_name}"
    params.load(f"{model_path}_params.json")
    model = build_model(params.model_type, mc_dropout=params.enable_mc_dropout)
    model.load_weights(f"{model_path}.h5")
    return model


def plot_pred_vs_true(files, model):
    input_data, y = format_data(files)
    real = y[:, 0]
    pred, pred_uncert = predict(model, input_data)
    fig, ax = plt.subplots()
    ax.errorbar(real, pred, yerr=pred_uncert, fmt='o', alpha=0.5)
    ax.set_xlabel("Real")
    ax.set_ylabel("Predicted")
    ax.set_title("Real vs Predicted Model Results")
    ax.set_xlim(params.curve_range)
    ax.set_ylim(params.curve_range)
    line = np.arange(params.curve_range[0], params.curve_range[1])
    ax.plot(line, line, color="black", alpha=0.5)


def plot_pred_ztf_max(files, model, save=False, save_dir=""):
    input_data, y = format_data(files)
    pred, pred_uncert = predict(model, input_data)
    real = y[:, 0]
    for i, file in enumerate(files):
        df = pd.read_csv(f"./TESS_data/processed_curves/{file}", index_col="relative_time")
        tess_name = file.split("_")[1]
        meta = get_curve_meta(tess_name)
        fig, ax = plt.subplots()
        display_all_passbands(df, meta, xlim=[-30, 70], ax=ax, title_info=format_title(meta))
        ax.axvline(pred[i], color="blue", linestyle="--", label="Predicted Max")
        ax.axvline(real[i], color="black", linestyle="--")
        if save:
            fig.savefig(save_dir + tess_name, bbox_inches="tight")
            plt.close(fig)



