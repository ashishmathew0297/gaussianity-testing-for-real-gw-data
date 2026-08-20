import os
import ast
import math as math
import pycbc as pycbc
import numpy as np
import warnings as warnings
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import lilliefors
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from matplotlib.ticker import ScalarFormatter
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn import metrics
from typing import Literal
from dotenv import find_dotenv, load_dotenv
from gwpy.table import GravitySpyTable
from numpy.typing import NDArray

warnings.filterwarnings('ignore')

def calculate_sample_statistics(y_values: list, scaler_type: Literal['minmax', 'standard', 'robust','none']="standard", threshold: float=0.05) -> dict:
    '''
    This function uses the input glitch TimeSeries sample in pycbc form to calculate and return a list of the following

    - Shapiro-Wilks statistic
    - Shapiro-Wilks p-value
    - Kolmogorov-Smirnov statistic
    - Kolmogorov-Smirnov p-value
    - Anderson-Darling statistic
    - Anderson-Darling critical values
    - Anderson-Darling significance levels

    Inputs:
    - `glitch_timeseries`: TimeSeries object of the glitch

    Output:
    - **data_df**: a list containing the following
        - 'shapiro_statistic': Shapiro-Wilks statistic of the sample amplitudes
        - 'shapiro_pvalue': Shapiro-Wilks p-value of the sample amplitudes
        - 'ks_pvalue': Kolmogorov-Smirnov p-value of the sample amplitudes
        - 'ad_statistic': Anderson-Darling statistic
        - 'ad_critical_values': Critical values for the Anderson Darling statistic
        - 'ad_significance_level': Significance level for the Anderson Darling statistic
        - 'kurtosis': Kurtosis of the glitch amplitude values
        - 'skew': Skew of the glitch amplitude values
    '''

    # np.random.seed(42)
    # np.random.seed()

    if scaler_type == 'minmax':
        scaler = MinMaxScaler(feature_range=(-4,4))
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'none':
        scaler = None
    else:
        raise ValueError("Invalid scaler type. Please choose from 'minmax', 'standard', 'robust', or 'none'.")

    if scaler is not None:
        scaled_y_values = list(scaler.fit_transform(np.array(y_values).reshape(-1,1))[:,0])
    else:
        scaled_y_values = y_values

    # =================== Shapiro-Wilks Test ===================

    sw_statistic = stats.shapiro(scaled_y_values)

    # =================== Two-Sample Kolmogorov-Smirnov Test ===================

    # The Kolmogorov Smirnov statistic needs to be applied to a scaled
    # version of our data to work properly since it is a distance-based
    # metric
    # ks_statistic = stats.ks_2samp(scaled_y_values, stats.norm.rvs(size=len(y_values)))
    ks_statistic = stats.ks_1samp(scaled_y_values, stats.norm.cdf)

    # =================== Anderson-Darling Test ===================
    # for our use case we consider a significance level of 5%
    ad_statistic = stats.anderson(scaled_y_values, dist='norm')

    # =================== Jarque-Bera Test ===================
    jb_statistic = stats.jarque_bera(scaled_y_values)

    # =================== Cramer-von Mises Test ===================

    cvm_statistic = stats.cramervonmises(scaled_y_values, stats.norm.cdf)

    # ==================== Normal Test (D'Agostino and Pearson) ===================

    normal_test = stats.normaltest(scaled_y_values)

    # =================== Lilliefors Test ===================
    lilliefors_test = lilliefors(scaled_y_values)

    # =================== Skew and Kurtosis ===================

    kurtosis = stats.kurtosis(scaled_y_values, fisher=False)
    skew = stats.skew(scaled_y_values)

    # Later Work if needed: KL Divergence


    return {
        "shapiro_statistic": sw_statistic.statistic,
        "shapiro_pvalue": sw_statistic.pvalue,
        "shapiro_prediction": 1 if sw_statistic.pvalue < threshold else 0,
        "ks_statistic": ks_statistic.statistic,
        "ks_pvalue": ks_statistic.pvalue,
        "ks_prediction": 1 if ks_statistic.pvalue < threshold else 0,
        "ad_statistic": ad_statistic.statistic,
        "ad_critical_values": ad_statistic.critical_values.tolist(),
        "ad_significance_level": ad_statistic.significance_level.tolist(),
        "ad_prediction": 1 if ad_statistic.statistic > ad_statistic.critical_values[np.where(np.array(ad_statistic.significance_level) <= 5)[0][0]] else 0,
        "cvm_statistic": cvm_statistic.statistic,
        "cvm_pvalue": cvm_statistic.pvalue,
        "cvm_prediction": 1 if cvm_statistic.pvalue < threshold else 0,
        "jb_statistic": jb_statistic.statistic,
        "jb_pvalue": jb_statistic.pvalue,
        "jb_prediction": 1 if jb_statistic.pvalue < threshold else 0,
        "normal_test_statistic": normal_test.statistic,
        "normal_test_pvalue": normal_test.pvalue,
        "normal_test_prediction": 1 if normal_test.pvalue < threshold else 0,
        "lilliefors_statistic": lilliefors_test[0],
        "lilliefors_pvalue": lilliefors_test[1],
        "lilliefors_prediction": 1 if lilliefors_test[1] < threshold else 0,
        "kurtosis": kurtosis,
        "skew": skew
    }

def get_section_statistics(data: pd.DataFrame, stat_test: Literal["Shapiro", "JB", "KS", "Anderson", "CVM", "normal_test", "Lilliefors"]="Shapiro", section_duration_seconds: float=1) -> list:
    '''
    A function to calculate one of the following:
    - Shapiro-Wilks Test p-values
    - Kolmogorov-Smirnov Test p-value
    - Anderson-Darling Statistics
    
    for sections of a sample glitch.

    Input:
    - **data:** A **single row** of glitch information. Must contain ['t', 'whitened_y']
    - **stat_test:** The test being performed on the sections (values=["Shapiro", "KS", "Anderson"]). Default="Shapiro".
    - **section_size_seconds:** The number of sections (in seconds) being studied. The accepted values range from 0 (exclusive) to 1 with a maximum precision of 4. Default=1 second.

    Display: A plot of
    - The glitch sample timeseries with sections highlighted to show the concerned statistics for each section.
    - A Q-Q plot of the whole sample

    Output:
      - **section_statistics:** A list of test results in relation to each of the sections of the dataset.
    '''

    section_info = []
    section_statistic = {}
    sample_length = len(data['whitened_y'])

    # Section size (in seconds) rounded to 5 places
    section_duration_seconds = round(section_duration_seconds, 5)
    
    # Using the sample timeframe in seconds, get section size
    # in terms of sampling rate
    # Checks:
    # 1. If section duration is less than or equal to the sample length in seconds
    # 2. If section duration is greater than 0
    # If both conditions are satisfied, calculate the section size
    # else use the whole sample as the section
    if section_duration_seconds <= sample_length/4096 and section_duration_seconds > 0:
        section_size = int(math.floor(sample_length) * section_duration_seconds)
    else:
        section_size = sample_length

    print(f"{stat_test} Statistics")
    print("====================")

    # =================== Section-wise Statistics Calculation ===================

    for i in range(0, len(data['whitened_y']+1), section_size):

        y = data['whitened_y'][i:i+section_size]
        t = np.array(data['t'])[i:i+section_size]

        # Calculating the section statistics
        if len(y) > 0:
            if stat_test == "Shapiro":
                section_statistic = stats.shapiro(y)._asdict()
            elif stat_test == "KS":
                # scaler = MinMaxScaler(feature_range=(-4,4))
                scaler = StandardScaler()
                section_statistic = stats.ks_2samp(list(scaler.fit_transform(y.reshape(-1,1))[:,0]), stats.norm.rvs(size=len(y)))._asdict()
            elif stat_test == "Anderson":
                section_statistic = stats.anderson(y, dist='norm')._asdict()

            if not np.isnan(section_statistic['statistic']):
                section_info.append({"whitened_y":y, "t":t, "section_statistic":section_statistic})

    return section_info

def generate_confusion_matrix(
        data: pd.DataFrame,
        stat_test: Literal["Shapiro", "JB", "KS", "Anderson", "CVM", "normal_test", "Lilliefors"]="Shapiro") -> NDArray:
    '''
    Generate a confusion matrix for the performance of the relevant statistical tests on the signal sample. The statistical tests being considered are
    - Shapiro-Wilks Test
    - Kolmogorov-Smirnov Test
    - Anderson-Darling Test

    Inputs:
    - `data`: The dataset of IFO signal information being studied.
    - `stat_test`: The statistical test being considered.

    Output:
    - Confusion matrix for the concerned statistic.
    '''

    test_to_column = {
        "Shapiro": "shapiro_prediction",
        "JB": "jb_prediction",
        "KS": "ks_prediction",
        "normal_test": "normal_test_prediction",
        "CVM": "cvm_prediction",
        "Anderson": "ad_prediction",
        "Lilliefors": "lilliefors_prediction"
    }

    prediction_column = test_to_column[stat_test]
    cm = metrics.confusion_matrix(data["glitch_present"], data[prediction_column], labels=[1,0])
    
    # if stat_test == "Shapiro":
    #     cm = metrics.confusion_matrix(data["glitch_present"],data["shapiro_prediction"],labels=[1,0])
    # if stat_test == "JB":
    #     cm = metrics.confusion_matrix(data["glitch_present"],data["jb_prediction"],labels=[1,0])
    # if stat_test == "KS":
    #     cm = metrics.confusion_matrix(data["glitch_present"],data["ks_prediction"],labels=[1,0])
    # if stat_test == "normal_test":
    #     cm = metrics.confusion_matrix(data["glitch_present"],data["normal_test_prediction"],labels=[1,0])
    # if stat_test == "CVM":
    #     cm = metrics.confusion_matrix(data["glitch_present"],data["cvm_prediction"],labels=[1,0])
    # if stat_test == "Anderson":
    #     cm = metrics.confusion_matrix(data["glitch_present"],data["ad_prediction"],labels=[1,0])
    # if stat_test == "Lilliefors":
    #     cm = metrics.confusion_matrix(data["glitch_present"],data["lilliefors_prediction"],labels=[1,0])
    
    return cm

def generate_evaluation_metrics(confusion_matrix):
    """
    Prints evaluation metrics given a confusion matrix.
    
    Parameters:
    confusion_matrix (list of list of int or scipy sparse matrix): 2x2 confusion matrix
    """
    # Convert scipy sparse matrix to dense if necessary
    if issparse(confusion_matrix):
        confusion_matrix = confusion_matrix.toarray()
    
    TP, FN, FP, TN = confusion_matrix[0][0], confusion_matrix[0][1], confusion_matrix[1][0], confusion_matrix[1][1]
    
    def safe_divide(num, denom):
        return num / denom if denom != 0 else 0
    
    # Calculate metrics
    accuracy = safe_divide(TP + TN, TP + TN + FP + FN)
    recall = safe_divide(TP, TP + FN)
    specificity = safe_divide(TN, TN + FP)
    fpr = safe_divide(FP, TN + FP)
    fnr = safe_divide(FN, FN + TP)
    precision = safe_divide(TP, TP + FP)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    # Print metrics
    return accuracy, recall, specificity, fpr, fnr, precision, f1_score