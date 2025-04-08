# In this python program, the flare catalog(with cme) is used as the label source.
# To create the label, log scale flare intensity is used
import glob
import argparse
import os.path, os
import pandas as pd

# pd.options.mode.chained_assignment = None


# In this function, to create the label, the maximum intensity of flare between midnight to midnight
# and noon to noon with respective date is used.
def hourly_obs(df_fl: pd.DataFrame, img_dir, start, stop, class_type="bin"):

    # Datetime
    df_fl["start_time"] = pd.to_datetime(
        df_fl["start_time"], format="%Y-%m-%d %H:%M:%S"
    )

    # List to store intermediate results
    lis = []
    cols = ["Timestamp", "goes_class", "label"]

    for year in range(start, stop + 1):
        for month in range(1, 13):
            for day in range(1, 32):
                dir = img_dir + f"{year}/{month:02d}/{day:02d}/*.jpg"
                files = sorted(glob.glob(dir))

                for file in files:
                    window_start = pd.to_datetime(
                        file.split("HMI.m")[1][:-4], format="%Y.%m.%d_%H.%M.%S"
                    )
                    window_end = window_start + pd.Timedelta(
                        hours=23, minutes=59, seconds=59
                    )

                    emp = (
                        df_fl[
                            (df_fl.start_time > window_start)
                            & (df_fl.start_time <= window_end)
                        ]
                        .sort_values("goes_class", ascending=False)
                        .head(1)
                        .squeeze(axis=0)
                    )
                    if pd.Series(emp.goes_class).empty:
                        ins = "FQ"
                        target = 0
                    else:
                        ins = emp.goes_class

                        if class_type == "bin":
                            if ins >= "M1.0":  # FQ and A class flares
                                target = 1
                            else:
                                target = 0
                        elif class_type == "multi":

                            if ins >= "M1.0":  # FQ and A class flares
                                target = 3
                            elif ins >= "C1.0":
                                target = 2
                            elif ins >= "B1.0":
                                target = 1
                            else:
                                target = 0

                    lis.append([window_start, ins, target])

    df_out = pd.DataFrame(lis, columns=cols)

    # df_out['Timestamp'] = pd.to_datetime(df_out['Timestamp'], format='%Y-%m-%d %H:%M:%S')
    df_out["Timestamp"] = df_out["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    return df_out


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
            columns=["Timestamp", "goes_class", "label"],
        )


if __name__ == "__main__":

    # Load Original source for Goes Flare X-ray Flux
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="/workspace/data/", help="Path to data folder"
    )
    parser.add_argument(
        "--project_path",
        type=str,
        default="/workspace/Project/baseline_fulldisk",
        help="Path to project folder",
    )
    parser.add_argument(
        "--start", type=int, default="2011", help="start time of the dataset"
    )
    parser.add_argument(
        "--end", type=int, default="2014", help="end time of the dataset"
    )
    args = parser.parse_args()
    
    df_fl = pd.read_csv(args.data_path + 'catalog/sdo_era_goes_flares_integrated_all_CME_r1.csv', usecols = ['start_time', 'goes_class'])

    savepath = os.getcwd()

    # Calling functions in order
    df_res = hourly_obs(
        df_fl=df_fl,
        img_dir=args.data_path + "hmi_jpgs_512/",
        start=args.start,
        stop=args.end,
        class_type="bin",
    )
    split_dataset(df_res, savepath=savepath, class_type="bin")
