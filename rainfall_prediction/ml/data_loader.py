

import pandas as pd
import numpy as np

FEATURE_NAMES = [
    "pressure", "maxtemp", "temperature", "mintemp",
    "dewpoint", "humidity", "cloud", "sunshine",
    "winddirection", "windspeed", "rain_today",
]

FEATURE_LABELS = {
    "pressure":      "Pressure (hPa)",
    "maxtemp":       "Max Temperature (°C)",
    "temperature":   "Temperature (°C)",
    "mintemp":       "Min Temperature (°C)",
    "dewpoint":      "Dew Point (°C)",
    "humidity":      "Humidity (%)",
    "cloud":         "Cloud Cover (%)",
    "sunshine":      "Sunshine (hrs)",
    "winddirection": "Wind Direction (°)",
    "windspeed":     "Wind Speed (km/h)",
    "rain_today":    "Rain Today",
}

FEATURE_EXPECTED_CORR = {
    "pressure":      -1,
    "maxtemp":       -1,
    "temperature":   -1,
    "mintemp":       +1,
    "dewpoint":      +1,
    "humidity":      +1,
    "cloud":         +1,
    "sunshine":      -1,
    "winddirection":  0,
    "windspeed":     +1,
    "rain_today":    +1,
}


def load_dataset(csv_path="Rainfall.csv"):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    if "temparature" in df.columns:
        df = df.rename(columns={"temparature": "temperature"})
    df["winddirection"] = df["winddirection"].fillna(df["winddirection"].median())
    df["windspeed"]     = df["windspeed"].fillna(df["windspeed"].median())

    df["rain_today"] = (df["rainfall"].str.strip().str.lower() == "yes").astype(int)
    df["rain_tomorrow"] = df["rain_today"].shift(-1)
    df = df.dropna(subset=["rain_tomorrow"])
    df["rain_tomorrow"] = df["rain_tomorrow"].astype(int)

    X = df[FEATURE_NAMES].values.astype(float)
    y = df["rain_tomorrow"].values

    return X, y, df


def compute_feature_stats(X, y):
    from scipy import stats as sp_stats
    stats = {}
    for i, fname in enumerate(FEATURE_NAMES):
        col = X[:, i]

        # Wind direction is circular — skip correlation check
        if fname == "winddirection":
            corr, pval = 0.0, 1.0
        else:
            corr, pval = sp_stats.pointbiserialr(col, y)

        expected = FEATURE_EXPECTED_CORR[fname]
        # Allow tiny correlations (< 0.03) to pass regardless of sign
        direction_ok = (expected == 0) or (np.sign(corr) == np.sign(expected)) or (abs(corr) < 0.03)

        stats[fname] = {
            "label":        FEATURE_LABELS[fname],
            "mean":         round(float(col.mean()), 3),
            "std":          round(float(col.std()),  3),
            "min":          round(float(col.min()),  3),
            "max":          round(float(col.max()),  3),
            "correlation":  round(float(corr),       4),
            "p_value":      round(float(pval),        6),
            "significant":  bool(pval < 0.05),
            "direction_ok": bool(direction_ok),
            "expected_dir": "+" if expected > 0 else ("−" if expected < 0 else "~"),
        }

    return stats
