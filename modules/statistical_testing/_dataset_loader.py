import os
import time
import tqdm
import math as math
import pycbc as pycbc
import numpy as np
import warnings as warnings
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from gwpy.signal import filter_design
from matplotlib.ticker import ScalarFormatter
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from typing import Literal
from dotenv import find_dotenv, load_dotenv
from gwpy.table import GravitySpyTable
from numpy.typing import NDArray

from ._statistics import calculate_sample_statistics

warnings.filterwarnings('ignore')

def get_TimeSeries(
    gps_time: float,
    gps_end_time: float=0,
    whitening_tw: int=10,
    srate: int=4096,
    ifo='L1',
    channel: str = "L1:DCS-CALIB_STRAIN_C01") -> str:
    '''
    This function fetches data from the GWOSC TimeSeries API and stores them in "./glitch_timeseries_data" corresponding to the sample if not already present.

    Inputs:
    - `gpstime`: The GPS time of the sample
    - `gps_end_time`: The end time of the sample. Default = 0 (not provided). If provided, the function will fetch data from gps_time to gps_end_time
    - `whitening_tw`: Time window to be taken into consideration on either side of the glitch for whitening. The final sample returned will contain the central 1 second of the sample.
            Default = 5 seconds 
    - `srate`: The sampling rate. Default= 4096
    - `ifo`: The interferometer being studied. Default=L1 (LIGO Livingston Observatory)
    - `channel`: The channel to fetch data from. Default="L1:DCS-CALIB_STRAIN_C01"
    '''

    timeseries_folder = "./timeseries_data/"
    os.makedirs(timeseries_folder, exist_ok=True)

    if not gps_end_time:
        filename = f"sample_{ifo}_{gps_time}_white{whitening_tw}.h5"
        start_time = gps_time - whitening_tw
        end_time = gps_time + whitening_tw
    else:
        start_time, end_time = gps_time, gps_end_time # End time already provided
        filename = f"sample_{ifo}_cleanseg_{start_time}_{end_time}.h5"

    timeseries_filepath = os.path.join(timeseries_folder, filename)

    # Loading and saving the TimeSeries for the given sample
    if not os.path.isfile(timeseries_filepath):
        unwhitened_noise = TimeSeries.fetch(
            channel,
            start_time,
            end_time,
            verbose=True)
        
        unwhitened_noise = unwhitened_noise.resample(srate)

        unwhitened_noise.write(timeseries_filepath)
    
    return timeseries_filepath


def calculate_transforms(sample: TimeSeries, function_name: Literal["q_transform", "q_gram", "fft"]="q_transform") -> tuple[NDArray, float]:
    '''
    This function returns the q-transform, q-gram and FFT of the sample

    Inputs:
    - `sample`: The whitened timeseries sample

    Outputs:
    - `q_scan`: q-scan of the sample
    - `q_gram`: q-gram of the sample
    - `fft`: FFT of the sample
    '''
    if function_name == "q_gram":
        start_time = time.process_time()
        q_scan = sample.q_gram(qrange=[4,64], frange=[10, 2048])
        end_time = time.process_time()
    elif function_name == "fft":
        start_time = time.process_time()
        q_scan = sample.psd(fftlength=0.015625, overlap=0.0078125)
        end_time = time.process_time()
    else:
        start_time = time.process_time()
        q_scan = sample.q_transform(qrange=[4,64], frange=[10, 2048], tres=0.002, fres=0.5, whiten=False)
        end_time = time.process_time()
    return q_scan, end_time - start_time


def process_dataset(
    data: pd.DataFrame,
    datatype: Literal['glitch', 'clean'],
    whitening_tw: int=10,
    observation_tw: float = 1,
    srate=4096,
    ifo='L1',
    bandpass: bool=False,
    low_freq: int=10,
    high_freq: int=250,
    filter_before_whitening: bool=True,
    output_file:str="",
    output_folder:str="./outputs",
    scaler_type: Literal["minmax", "standard", "robust"]="standard",
    postfix:str="")-> None:

    '''
    Unified function to perform the statistical tests on the glitch and clean TimeSeries samples and return a dataset with the relevant information appended 

    Inputs:
    - `data`: Pandas dataframe containing the data to process
    - `datatype`: Either 'glitch' or 'clean' to specify the type of data being processed
    - `gpsTimeKey`: The key value for GPS time in the dataset (for glitch data)
    - `whitening_tw`: Time window for whitening (for glitch data)
    - `observation_tw`: Observation time window (for glitch data)
    - `srate`: Sampling rate
    - `ifo`: Interferometer identifier
    - `segment_duration_seconds`: Duration of each segment (only used for clean data)
    - `n_samples`: Number of random samples to select (for clean data, 0 = all)
    - `bandpass`: Whether to apply bandpass filter
    - `low_freq`: Lower frequency for bandpass
    - `high_freq`: Upper frequency for bandpass
    - `output_file`: Custom output filename (without extension)
    - `postfix`: Postfix to add to output files (e.g., frequency band identifier)

    Output: The original input dataset concatenated with the following
    - 'whitened_y': Amplitude values of the whitened glitch timeseries
    - 't': Time values of the whitened glitch timeseries,
    - 'q_transform': Q-transform of the whole glitch sample (1 second removed at either end to account for border effects)
    - 'Shapiro-Wilk statistic': Shapiro statistic of the sample amplitudes
    - 'Shapiro-Wilk p-value': Shapiro p-value of the sample amplitudes
    - 'Shapiro-Wilk prediction': The prediction made based on the Shapiro-Wilks p-value of sample amplitudes
    - 'Kolmogorov-Smirnov Statistic': Kolmogorov-Smirnov statistic of the sample amplitudes
    - 'Kolmogorov-Smirnov p-value': Kolmogorov-Smirnov p-value of the sample amplitudes
    - 'Kolmogorov-Smirnov prediction': The prediction made based on the Kolmogorov-Smirnov p-value of sample amplitudes
    - 'Anderson-Darling statistic': Anderson-Darling statistic
    - 'Anderson-Darling critical values': Critical values for the Anderson Darling statistic
    - 'Anderson-Darling significance level': Significance level for the Anderson Darling statistic
    - 'Anderson-Darling prediction': The prediction made based on the Anderson-Darling statistic
    - 'Kurtosis': Kurtosis of the glitch amplitude values
    - 'Skew': Skew of the glitch amplitude values
    '''

    base_filename = output_file or f"{datatype}_statistical_results"
    postfix = f"_{postfix}" if postfix else ""
    output_file = f"{output_folder}/{base_filename}{postfix}.csv"

    # output_df = pd.DataFrame(columns=['id', 'ifo', 'GPStime', 'start_time', 'end_time', 'label'])
    output_df = pd.DataFrame()

    # Creating output directory and file if not already present
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if os.path.isfile(output_file):
        output_df = pd.read_csv(output_file)

    # Only for clean signals
    clean_segment_size = int(srate * observation_tw)

    # Only for glitch samples
    crop_start = int(srate * (whitening_tw - (observation_tw/2)))
    crop_end =  - crop_start if crop_start != 0 else None

    # ascii=" ▏▎▍▌▋▊▉"
    # ascii=" ▁▂▃▄▅▆▇█"

    for i in tqdm.tqdm(range(len(data)), desc=f"Processing {datatype} samples", ascii=" ▖▘▝▗▚▞█", miniters=10000):

        row_dict = data.iloc[i].to_dict()

        # skip current row if already processed and present in output file
        if not output_df.empty and row_dict['id'] in output_df['id'].values:
            continue
        
        # Skip row if timeseries file is not present
        if 'timeseries_filepath' not in row_dict or not row_dict['timeseries_filepath'] or not os.path.isfile(row_dict['timeseries_filepath']):
            print(f"Timeseries file not found for id {row_dict['id']}")
            continue

        # Try clause is the normal program flow
        # Except clause skips the current iteration and enters zero values
        # if TimeSeries fails to load
        try:

            if not os.path.isfile(row_dict['timeseries_filepath']):
                raise FileNotFoundError(f"File {row_dict['timeseries_filepath']} not found.")
            unwhitened_noise = TimeSeries.read(row_dict['timeseries_filepath'])
            
            # Band pass between 50 to 250 Hz is ideal for Scattered Light glitches
            if bandpass and filter_before_whitening:
                bp = filter_design.bandpass(low_freq, high_freq, srate)
                unwhitened_noise = unwhitened_noise.filter(bp)
            
            whitened_noise = unwhitened_noise.whiten(4, 2)

            if bandpass and not filter_before_whitening:
                bp = filter_design.bandpass(low_freq, high_freq, srate)
                whitened_noise = whitened_noise.filter(bp)

            if datatype == 'glitch':
                whitened_noise = whitened_noise[crop_start:crop_end]
                whitened_y = whitened_noise.value

                supplemental_glitch_data = calculate_sample_statistics(whitened_y, scaler_type=scaler_type)
                # supplemental_glitch_data['id'] = row['id']
                # supplemental_glitch_data['ifo'] = row.get('ifo', ifo)
                # supplemental_glitch_data['GPStime'] = row.get('GPStime', 0)
                # supplemental_glitch_data['label'] = row.get('label', 'Unknown')
                # supplemental_glitch_data['timeseries_filepath'] = row['timeseries_filepath']
                
                # ADDITIONAL COLUMNS ADDED TO MAINTAIN UNIFORMITY WITH CLEAN DATA
                supplemental_glitch_data['start_time'] = 0
                supplemental_glitch_data['end_time'] = 0
                supplemental_glitch_data['glitch_present'] = 1
                row_dict.update(supplemental_glitch_data)


                result_df = pd.DataFrame([row_dict])
            else:
                # Getting rid of border effects before splitting into segments
                whitened_noise = whitened_noise[int(srate * 1):-int(srate * 1)]

                if clean_segment_size >= len(whitened_noise):
                    continue

                whitened_samples = []
                for j in range(clean_segment_size, len(whitened_noise) + 1, clean_segment_size):
                    sample = whitened_noise[j - clean_segment_size:j]

                    # Only accept samples that are of the exact segment size
                    if len(sample) == clean_segment_size:

                        sample_data = sample.value
                        sample_times = sample.times

                        segment_data = calculate_sample_statistics(sample_data, scaler_type=scaler_type)
                        segment_data['start_time'] = sample_times[0]
                        segment_data['end_time'] = sample_times[-1]

                        if postfix:
                            timeseries_filename = f"whitened_sample_{ifo}_clean_{segment_data['start_time']}_{segment_data['end_time']}_segmentsize_{observation_tw}_{postfix}.h5"
                        else:
                            timeseries_filename = f"whitened_sample_{ifo}_clean_{segment_data['start_time']}_{segment_data['end_time']}_segmentsize_{observation_tw}.h5"

                        # saving the whitened timeseries of the sample for later use in analysis
                        timeseries_filepath = f"./timeseries_data/{timeseries_filename}"
                        if not os.path.isfile(timeseries_filepath):
                            print(f"Saving {timeseries_filepath}")
                            sample.write(timeseries_filepath)
                                
                        segment_data['timeseries_filepath'] = timeseries_filepath
                        whitened_samples.append(segment_data)
                        
                if not whitened_samples:
                    print(f"No valid segments created for id {row_dict['id']}", flush=True)
                    continue

                result_df = pd.DataFrame(whitened_samples)
                result_df["id"] = row_dict['id']
                result_df["ifo"] = row_dict.get('ifo', ifo)
                result_df["label"] = "Clean_Signal"
                result_df["glitch_present"] = 0
                result_df["snr"] = 0
                result_df["GPStime"] = 0
            
            # Write to output file
            if os.path.isfile(output_file):
                result_df.to_csv(output_file, mode='a', header=False, index=False)
            else:
                result_df.to_csv(output_file, mode='a', header=True, index=False)

            
        except KeyboardInterrupt:
            print("Keyboard Interrupt detected. Exiting...")
            break
        except Exception as e:
            print(f"Error processing {datatype} sample with id {row_dict['id']}: {e}")
            continue

    print(f"{datatype.capitalize()} data processing complete!")


def fetch_gspy_glitch_data(glitchtype: str) -> None:
    filepath = f"./gspy_glitches/gspy_{glitchtype}.csv"
    load_dotenv(find_dotenv())

    GRAVITYSPY_DATABASE_USER = os.getenv('GRAVITYSPY_DATABASE_USER')
    GRAVITYSPY_DATABASE_PASSWD = os.getenv('GRAVITYSPY_DATABASE_PASSWD')
    
    if not os.path.exists("./gspy_glitches"):
        os.makedirs("./gspy_glitches")
    if not os.path.exists(filepath):
        glitch_data = GravitySpyTable.fetch(
            "gravityspy",
            "glitches",
            selection=f"ml_label={glitchtype}"
        ).to_pandas()
        glitch_data.to_csv(filepath, index=False)