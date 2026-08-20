import os
import argparse
import pandas as pd

from globals import *

if not os.path.isdir("./temp"):
    os.mkdir("./temp")

parser = argparse.ArgumentParser(description="Split the data into clean and glitch samples for statistical testing")
parser.add_argument("--num_chunks", type=int, required=True, help="Number of chunks for parallel processing")
args = parser.parse_args()

glitches = pd.read_csv(glitch_times_file)
glitches = glitches[~glitches.duplicated(subset=["GPStime"], keep="first")]
glitches = glitches[glitches["label"] != "No_Glitch"]

clean_samples = pd.read_csv(clean_segments_file)

def save_chunks(dataset_size, datatype, num_chunks):
    chunk_size = dataset_size // num_chunks
    indices = [i * chunk_size for i in range(num_chunks)] + [dataset_size]
    
    if not os.path.isfile(f"./temp/splits_{datatype}.txt"):
        with open(f"./temp/splits_{datatype}.txt", "w") as f:
            for i in range(num_chunks):
                f.write(f"{i},{indices[i]},{indices[i+1]}\n")


# save start and end indices for each chunk to a txt.
save_chunks(len(clean_samples), "clean", args.num_chunks)
save_chunks(len(glitches), "glitch", args.num_chunks)

print("Data Splitting Complete") 