#!/usr/bin/env python3
"""Multiple Linear Regression model training script.

Trains one MLR model per BMC engine and saves trained model dict to a pickle file.

"""

import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

BMC_ENGINES = ["bmc2", "bmc3", "bmc3g", "bmc3r", "bmc3u", "bmc3s", "bmc3j"]
FEATURE_COLS = ["Var", "Cla", "Conf", "Learn"]
TARGET_COL = ["Time"]


def train(
    train_data_path: Path,
    bmc_engines: list[str],
    feature_cols: list[str],
    target_col: list[str],
) -> dict[str, LinearRegression]:
    X_dict: dict[str, pd.DataFrame] = {}
    y_dict: dict[str, pd.DataFrame] = {}
    model_dict: dict[str, LinearRegression] = {}
    for bmc_engine in bmc_engines:
        X = pd.DataFrame({col: [] for col in feature_cols})
        y = pd.DataFrame({col: [] for col in target_col})

        engine_train_data_path = train_data_path / bmc_engine
        csv_files = list(engine_train_data_path.glob("*.csv"))
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)

            #df['del_T'] = df['Time'].diff()
            #df = df.dropna(how='any')

            X = pd.DataFrame(pd.concat([X, df[feature_cols]], axis=0))
            y = pd.DataFrame(pd.concat([y, df[target_col]], axis=0))

        X_dict[bmc_engine] = X
        y_dict[bmc_engine] = y

        model_dict[bmc_engine] = LinearRegression()
        model_dict[bmc_engine].fit(X, y)

    return model_dict


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mlr.py",
        description="Trains one MLR model per BMC engine and saves trained model dict to a pickle file.",
    )
    parser.add_argument(
        "-d",
        "--train-data-path",
        default="../data/train_data_csv/",
        help="Path to training dir with bmc dirs, each bmc dir containing CSVs for circuits.",
        type=Path,
    )
    parser.add_argument(
        "-m",
        "--model-pkl-file",
        default="../data/model.pkl",
        help="Path to output pickle file where trained model dict is stored.",
        type=Path,
    )
    args = parser.parse_args()

    model_dict = train(args.train_data_path, BMC_ENGINES, FEATURE_COLS, TARGET_COL)

    with open(args.model_pkl_file, "wb") as pkl_file:
        pickle.dump(model_dict, pkl_file)


if __name__ == "__main__":
    main()
