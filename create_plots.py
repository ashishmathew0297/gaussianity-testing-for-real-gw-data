import os
import argparse
import warnings
import numpy as np
import pandas as pd
import tqdm

from modules import statistical_testing
from globals import *

parser = argparse.ArgumentParser(description="View Statistical Results for Glitch Detection")

parser.add_argument("--clean_output_file", type=str, default="./outputs/clean_statistical_results.csv", help="Path to the clean statistical results CSV file")
parser.add_argument("--glitch_output_file", type=str, default="./outputs/glitch_statistical_results.csv", help="Path to the glitch statistical results CSV file")
parser.add_argument("--latex_output_file", type=str, default="./outputs/test_conf_matrix_full.tex", help="Path to save the output LaTeX file")
parser.add_argument("--csv_output_file", type=str, default="./outputs/test_conf_matrix_full.csv", help="Path to save the output CSV file")
parser.add_argument("--output_folder", type=str, default="./outputs", help="Folder to save the output images")
parser.add_argument("--image_postfix", type=str, default="", help="Postfix to add to saved images")

args = parser.parse_args()
   
clean_df = pd.read_csv(args.clean_output_file, usecols=useful_cols)
glitch_df = pd.read_csv(args.glitch_output_file, usecols=useful_cols)
if args.image_postfix != "":
    postfix = "_" + args.image_postfix
else:
    postfix = args.image_postfix

def view_statistical_results(clean_df, glitch_df):

    combined_df = pd.concat([clean_df, glitch_df], ignore_index=True)

    # only considering the results for label counts greater than 100
    label_counts = combined_df['label'].value_counts()
    valid_labels = label_counts[label_counts >= 100].index.tolist()
    combined_df = combined_df[combined_df['label'].isin(valid_labels)]

    confusion_matrix_df = pd.DataFrame(columns=["Test","TP","FN","FP","TN", "Accuracy", "TPR", "TNR", "FPR", "FNR","Precision","F1 Score"])
    # Creating Confusion Matrices for each statistical test
    for test in tqdm.tqdm(["Shapiro", "JB", "KS", "Anderson", "CVM", "normal_test", "Lilliefors"], ascii=" ▖▘▝▗▚▞█"):
        statistical_testing.display_confusion_matrix(combined_df, test, save_img=False, output_folder=args.output_folder, filename = f"conf_matrix_{test}{postfix}.pdf")
        cm = statistical_testing.generate_confusion_matrix(combined_df, test)
        metrics = list(statistical_testing.generate_evaluation_metrics(cm))
        cm = cm.flatten()
        confusion_matrix_df = pd.concat([confusion_matrix_df, pd.DataFrame([[test] + cm.tolist() + metrics], columns=confusion_matrix_df.columns)], ignore_index=True)
    confusion_matrix_df.to_latex(args.latex_output_file, index=False, float_format="%.2f", escape=False)
    confusion_matrix_df.to_csv(args.csv_output_file, index=False, float_format="%.2f")

    # Creating Confusion Matrices for each glitch type
    for test in tqdm.tqdm(["Shapiro", "JB", "KS", "Anderson", "CVM", "normal_test", "Lilliefors"], ascii=" ▖▘▝▗▚▞█"):
        confusion_matrix_df = pd.DataFrame(columns=["Label","TP","FN","FP","TN", "Accuracy", "TPR", "TNR", "FPR", "FNR","Precision","F1 Score"])
        # valid_glitch_labels = [label for label in glitch_labels if label in valid_labels and label != 'Clean_Signal']
        valid_glitch_labels = [label for label in glitch_labels if label in valid_labels]
        
        for glitch_label in valid_glitch_labels:
            glitch_specific_df = combined_df[combined_df['label'] == glitch_label]
            cm = statistical_testing.generate_confusion_matrix(glitch_specific_df, test)
            metrics = list(statistical_testing.generate_evaluation_metrics(cm))
            cm = cm.flatten()
            confusion_matrix_df = pd.concat([confusion_matrix_df, pd.DataFrame([[glitch_label] + cm.tolist() + metrics], columns=confusion_matrix_df.columns)], ignore_index=True)
        confusion_matrix_df[["Label","TP","FN", "TN", "FP", "TPR", "FNR", "TNR", "FPR"]].to_latex(f"{args.output_folder}/{test}_results_glitchwise{postfix}.tex", index=False, float_format="%.2f", escape=False)
        confusion_matrix_df[["Label","TP","FN", "TN", "FP", "TPR", "FNR", "TNR", "FPR"]].to_csv(f"{args.output_folder}/{test}_results_glitchwise{postfix}.csv", index=False, float_format="%.2f")

    # # Generating ROC Curves for each statistical test
    # for test in tqdm.tqdm(["Shapiro", "JB", "KS", "Anderson", "CVM", "normal_test", "Lilliefors"], ascii=" ▖▘▝▗▚▞█"):
    #     statistical_testing.display_auc_roc(combined_df, test, save_img=True)
    #     statistical_testing.display_auc_roc(combined_df, test, thresholds= np.linspace(0.01,1.0, 100) , save_img=True)


if __name__ == "__main__":
    view_statistical_results(clean_df, glitch_df)

