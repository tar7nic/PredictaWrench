import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

LOW_INFO_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]

ROLLING_WINDOW = 5

RUL_CLIP = 125


def load_clean():
    train = pd.read_csv(os.path.join(DATA_DIR, "train_clean.csv"))
    test  = pd.read_csv(os.path.join(DATA_DIR, "test_clean.csv"))
    return train, test


def drop_low_info(df):
    cols_to_drop = [c for c in LOW_INFO_SENSORS if c in df.columns]
    return df.drop(columns=cols_to_drop)


def get_active_sensors(df):
    non_sensor_cols = {"unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3", "RUL"}
    return [c for c in df.columns if c not in non_sensor_cols]


def add_rolling_features(df, sensors):
    df = df.sort_values(["unit", "cycle"]).copy()
    for sensor in sensors:
        df[f"{sensor}_roll_mean"] = (
            df.groupby("unit")[sensor]
            .transform(lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean())
        )
        df[f"{sensor}_roll_std"] = (
            df.groupby("unit")[sensor]
            .transform(lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).std().fillna(0))
        )
    return df


def clip_rul(df):
    df = df.copy()
    df["RUL"] = df["RUL"].clip(upper=RUL_CLIP)
    return df


def build_feature_matrix(df, is_train=True):
    df = drop_low_info(df)
    sensors = get_active_sensors(df)
    df = add_rolling_features(df, sensors)
    if is_train:
        df = clip_rul(df)
    return df


def get_feature_columns(df):
    non_feature = {"unit", "cycle", "RUL"}
    return [c for c in df.columns if c not in non_feature]


def run():
    print("Loading cleaned data...")
    train, test = load_clean()
    print(f"  Train shape before engineering: {train.shape}")
    print(f"  Test  shape before engineering: {test.shape}")

    print(f"Dropping low-information sensors: {LOW_INFO_SENSORS}")
    print(f"Computing rolling features (window={ROLLING_WINDOW})...")
    print(f"Clipping RUL at {RUL_CLIP} for piecewise linear degradation...")

    train_eng = build_feature_matrix(train, is_train=True)
    test_eng  = build_feature_matrix(test,  is_train=False)

    print(f"  Train shape after engineering : {train_eng.shape}")
    print(f"  Test  shape after engineering : {test_eng.shape}")

    feature_cols = get_feature_columns(train_eng)
    print(f"  Total feature columns         : {len(feature_cols)}")
    print(f"  RUL range after clipping      : [{train_eng['RUL'].min()}, {train_eng['RUL'].max()}]")

    out_train = os.path.join(DATA_DIR, "train_features.csv")
    out_test  = os.path.join(DATA_DIR, "test_features.csv")
    train_eng.to_csv(out_train, index=False)
    test_eng.to_csv(out_test,   index=False)

    print(f"Saved: {out_train}")
    print(f"Saved: {out_test}")
    print("✅")

    return train_eng, test_eng


if __name__ == "__main__":
    run()