import pandas as pd
import numpy as np
import os

COLUMN_NAMES = [
    "unit", "cycle",
    "op_setting_1", "op_setting_2", "op_setting_3",
    "s1", "s2", "s3", "s4", "s5",
    "s6", "s7", "s8", "s9", "s10",
    "s11", "s12", "s13", "s14", "s15",
    "s16", "s17", "s18", "s19", "s20", "s21"
]

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_TRAIN = os.path.join(DATA_DIR, "train_FD001.txt")
RAW_TEST  = os.path.join(DATA_DIR, "test_FD001.txt")
RAW_RUL   = os.path.join(DATA_DIR, "RUL_FD001.txt")


def load_raw(filepath):
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
        engine="python"
    )
    return df


def compute_train_rul(df):
    max_cycles = df.groupby("unit")["cycle"].max().rename("max_cycle")
    df = df.merge(max_cycles, on="unit")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df.drop(columns=["max_cycle"], inplace=True)
    return df


def load_test_rul(df_test):
    rul_values = pd.read_csv(RAW_RUL, header=None, names=["RUL_true"])
    rul_values["unit"] = rul_values.index + 1

    last_cycles = df_test.groupby("unit")["cycle"].max().reset_index()
    last_cycles = last_cycles.merge(rul_values, on="unit")

    df_test = df_test.merge(last_cycles[["unit", "RUL_true"]], on="unit", how="left")

    def assign_rul(group):
        unit_id = group["unit"].iloc[0]
        true_rul_at_end = rul_values.loc[rul_values["unit"] == unit_id, "RUL_true"].values[0]
        max_cycle = group["cycle"].max()
        group = group.copy()
        group["RUL"] = (max_cycle - group["cycle"]) + true_rul_at_end
        return group

    df_test = df_test.groupby("unit", group_keys=False).apply(assign_rul)
    df_test.drop(columns=["RUL_true"], inplace=True)
    return df_test


def null_check(df, name):
    nulls = df.isnull().sum().sum()
    print(f"  [{name}] nulls: {nulls}")


def run():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Loading raw CMAPSS FD001 files...")
    df_train = load_raw(RAW_TRAIN)
    df_test  = load_raw(RAW_TEST)

    print(f"  Train raw shape : {df_train.shape}")
    print(f"  Test  raw shape : {df_test.shape}")

    null_check(df_train, "train")
    null_check(df_test,  "test")

    print("Computing RUL labels for training set...")
    df_train = compute_train_rul(df_train)
    print(f"  RUL range in train: [{df_train['RUL'].min()}, {df_train['RUL'].max()}]")

    print("Assigning RUL to test set from RUL_FD001.txt...")
    df_test = load_test_rul(df_test)
    print(f"  RUL range in test : [{df_test['RUL'].min():.0f}, {df_test['RUL'].max():.0f}]")

    out_train = os.path.join(DATA_DIR, "train_clean.csv")
    out_test  = os.path.join(DATA_DIR, "test_clean.csv")
    df_train.to_csv(out_train, index=False)
    df_test.to_csv(out_test,  index=False)

    print(f"Saved: {out_train}")
    print(f"Saved: {out_test}")
    print("✅")

    return df_train, df_test


if __name__ == "__main__":
    run()