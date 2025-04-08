# In this python program, the flare catalog(with cme) is used as the label source.
# To create the label, log scale flare intensity is used
import glob
import argparse
import os.path, os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit


# In this function, to create the label, the maximum intensity of flare between midnight to midnight
# and noon to noon with respective date is used.
def hourly_obs(df_fl: pd.DataFrame, img_dir, start, stop, class_type="bin"):

    # Datetime
    df_fl["start_time"] = pd.to_datetime(
        df_fl["start_time"], format="%Y-%m-%d %H:%M:%S"
    )

    # List to store intermediate results
    lis = []
    cols = ["Timestamp", "goes_class", "label", "present"]

    for year in range(start, stop + 1):
        for month in range(1, 13):
            for day in range(1, 32):
                for hour in range(0, 24):

                    f_name = f"HMI.m{year}.{month:02d}.{day:02d}_{hour:02d}.00.00.jpg"
                    full_path = os.path.join(img_dir, f_name)

                    window_start = pd.to_datetime(
                            f"{year}{month:02d}{day:02}_{hour}0000", format="%Y%m%d_%H%M%S"
                        )
                    window_end = window_start + pd.Timedelta(
                        hours=23, minutes=59, seconds=59
                    )

                    if os.path.exists(full_path):

                        fl_class, label = find_max_intensity_reg(
                            window=df_fl[
                                (df_fl.start_time > window_start)
                                & (df_fl.start_time <= window_end)
                            ],
                            feature='goes_class'
                        )

                        lis.append([window_start, fl_class, label, 1])
                    else:
                        lis.append([window_start, "", "", 0])

    df_out = pd.DataFrame(lis, columns=cols)

    # df_out['Timestamp'] = pd.to_datetime(df_out['Timestamp'], format='%Y-%m-%d %H:%M:%S')
    df_out["Timestamp"] = df_out["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    return df_out


def find_max_intensity_cls(
        window: pd.DataFrame = None,
        feature: str = 'goes_class',
        threshold: str | int | float = 'M1.0') -> list:
    
    emp = window.sort_values("goes_class", ascending=False).head(1).squeeze(axis=0)
    
    if pd.Series(emp.loc[:, feature]).empty:
        fl_class = "FQ"
        label = 0
    else:
        fl_class = emp.loc[:, feature]

        if fl_class >= threshold:  # FQ and A class flares
            label = 1
        else:
            label = 0

    return fl_class, label


def find_max_intensity_reg(
        window: pd.DataFrame = None,
        feature: str = 'goes_class') -> list:

    emp = window.sort_values("goes_class", ascending=False).head(1).squeeze(axis=0)
    if pd.Series(emp.loc[:, feature]).empty:
        fl_class = ''
        label = 0
    else:
        fl_class = emp.loc[:, feature]
        label = np.log10(emp.loc[:, "goes_class_num"]) + 8

    return fl_class, label


# Creating time-segmented 4 tri-monthly partitions
def split_dataset(df, savepath="/", class_type="bin"):
    search_list = [["2011", "2012", "2013"], ["2014"]]
    for i in range(2):
        search_for = search_list[i]
        mask = (
            df["Timestamp"]
            .apply(lambda row: row[0:4])
            .str.contains("|".join(search_for))
        )
        partition = df[mask]
        print(partition["label"].value_counts())

        # Dumping the dataframe into CSV with label as Date and goes_class as intensity
        flag = "train" if i == 0 else "test"
        partition.to_csv(
            savepath + f"/24image_{class_type}_class_{flag}.csv",
            index=False,
            header=True,
            columns=df.columns
        )


def stratified_dataset(df, savepath, task_type="reg"):

    label_col = "label"

    # First split: 75% train_cal, 25% test
    sss_1 = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    train_cal_idx, test_idx = next(sss_1.split(df, df[label_col]))

    train_cal_df = df.iloc[train_cal_idx]
    test_df = df.iloc[test_idx]

    # Second split: 66.7% train, 33.3% cal → from the 75% train_cal
    sss_2 = StratifiedShuffleSplit(n_splits=1, test_size=1/3, random_state=0)
    train_idx, cal_idx = next(sss_2.split(train_cal_df, train_cal_df[label_col]))

    train_df = train_cal_df.iloc[train_idx]
    cal_df = train_cal_df.iloc[cal_idx]

    test_df.to_csv(
        os.path.join(savepath, f"24image_{task_type}_test.csv"),
        index=False,
        columns=df.columns
    )

    train_df.to_csv(
        os.path.join(savepath, f"24image_{task_type}_train.csv"),
        index=False,
        columns=df.columns
    )

    cal_df.to_csv(
        os.path.join(savepath, f"24image_{task_type}_train.csv"),
        index=False,
        columns=df.columns
    )


if __name__ == "__main__":

    data_path = "/workspace/data/"
    savepath = os.getcwd()
    start_time = "2011"
    end_time = "2014"

    df_fl = pd.read_csv(data_path + 'catalog/sdo_era_goes_flares_integrated_all_CME_r1.csv', usecols = ['start_time', 'goes_class'])

    # Calling functions in order
    df_res = hourly_obs(
        df_fl=df_fl,
        img_dir=data_path + "hmi_jpgs_512",
        start=start_time,
        stop=end_time,
        class_type="bin",
    )
    split_dataset(df_res, savepath=savepath, class_type="bin")
