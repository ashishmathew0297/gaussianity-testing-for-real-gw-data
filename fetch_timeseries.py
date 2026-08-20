import os
import sys
import warnings
import argparse
import pandas as pd

from modules import statistical_testing
from globals import *

parser = argparse.ArgumentParser(description="Process data and statistical tests for each frequency band")
parser.add_argument("--datatype", type=str, choices=["glitch", "clean"], required=True, help="Type of data to process (glitch or clean)")
parser.add_argument("--chunk_index", type=int, default=0, help="Index of the chunk (for parallel processing)")
parser.add_argument("--start_index", type=int, default=0, help="Starting index of the csv file to process")
parser.add_argument("--end_index", type=int, default=1500, help="Ending index of the csv file to process")
parser.add_argument("--whitening_tw", type=int, default=10, help="Time window at each side of the glitch for whitening")
args = parser.parse_args()

output_file = f"./temp/{args.datatype}_{args.chunk_index}.csv"

if os.path.isfile(output_file):
    data = pd.read_csv(output_file)
    time_col = "GPStime" if args.datatype == "glitch" else "start_time"
else:
    if args.datatype == "glitch":
        # data = pd.read_csv(glitch_times_file, usecols=["GPStime", "ifo", "label", "id"])
        data = pd.read_csv(glitch_times_file)
        data = data[~data.duplicated(subset=["GPStime"], keep="first")]
        data = data[data["label"] != "No_Glitch"]
        time_col = "GPStime"
    else:
        # data = pd.read_csv(clean_segments_file, usecols=["start_time", "end_time", "id"])
        data = pd.read_csv(clean_segments_file)
        time_col = "start_time"

    data["timeseries_filepath"] = ""
    data = data.iloc[args.start_index:args.end_index].reset_index(drop=True)

    # creating initial state of csv
    data.to_csv(output_file, index=False)

for index, row in data.iterrows():
    gps_time = row[time_col]
    gps_end_time = row.get("end_time", None)

    try:
        # check if timeseries_filepath is already populated to avoid redundant fetching
        if pd.notna(row["timeseries_filepath"]) and os.path.isfile(row["timeseries_filepath"]):
            continue
        timeseries_file_location = statistical_testing.get_TimeSeries(gps_time, gps_end_time, whitening_tw=args.whitening_tw)
        data.at[index, "timeseries_filepath"] = timeseries_file_location

        # saving every 10 iterations to ensure progress
        if index % 10 == 0:
            data.to_csv(output_file, index=False)
    except Exception as e:
        print(f"Error fetching sample with id {row['id']}: {e}", flush=True)
    
data.to_csv(output_file, index=False)
print(f"Data fetching complete for {args.datatype} chunk {args.chunk_index}")


    
