### Author Douwe Spaanderman - 30 September 2020 ###

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import argparse

# Currently use sys to get other script - in Future use package
import os
import sys
path_main = ("/".join(os.path.realpath(__file__).split("/")[:-2]))
sys.path.append(path_main + '/Classes/')
sys.path.append(path_main + '/Utils/')
from media_class import Medium, Supplement, GrowthMedium, Medium_one_hot, Supplement_one_hot, GrowthMedium_one_hot
from help_functions import mean, str_to_bool, str_none_check

def main_summary(data, Path=False, level="Highest", save=False, show=True):
    '''

    '''
    if Path != False:
        if not Path.endswith(".pkl"):
            Path = Path + "after_media.pkl"
        
        data = pd.read_pickle(Path)
        
    data = data.replace({"Growth": {1: "Yes", 0: "No"}})

    if level == "Highest":
        data = data[data["Disease_highest_level"] != "Unknown"]
        data = data.groupby(["Disease_highest_level", "Growth"]).size().reset_index(name="Count")

        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111)

        ax = sns.barplot(y='Count', x='Disease_highest_level', 
                        data=data, 
                        palette="colorblind",
                        hue='Growth')

        ax.set_xlabel('')
        ax.set_ylabel('Count', fontsize=20)
        ax.set_yticklabels([0, 200, 400, 600, 800, 1000],fontsize=14)
        ax.set_xticklabels(labels=data["Disease_highest_level"].unique(), rotation=45, ha='right', fontsize=14)
        plt.setp(ax.get_legend().get_texts(), fontsize='22')
        plt.setp(ax.get_legend().get_title(), fontsize='32')

    elif level == "Lowest":
        data = data[data["Disease_lowest_level"] != "Unknown"]
        data = data.groupby(["Disease_lowest_level", "Growth"]).size().reset_index(name="Count")

        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111)

        ax = sns.barplot(y='Count', x='Disease_lowest_level', 
                        data=data, 
                        palette="colorblind",
                        hue='Growth')

        ax.set_xlabel('')
        ax.set_ylabel('Count', fontsize=20)
        ax.set_yticklabels([0, 200, 400, 600, 800, 1000],fontsize=14)
        ax.set_xticklabels(labels=data["Disease_lowest_level"].unique(), rotation=45, ha='right', fontsize=14)
        plt.setp(ax.get_legend().get_texts(), fontsize='22')
        plt.setp(ax.get_legend().get_title(), fontsize='32')

    else:
        raise KeyError(f"Wierd disease level was passed: {level}. please provide either Highest or Lowest")

    if save != False:
        if not save.endswith("Figures/"):
            save += "Figures/"
        plt.savefig(save + "data_summary.png")
    if show == True:
        plt.show()
    if show == False and save == False:
        warnings.warn('Neither Save nor Show is True.. kinda worthless computer time but whatever')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Heatmap matrix of media and supplements in samples")
    parser.add_argument("Path", help="path to terra workspace filtered file after media.py")
    parser.add_argument("-l", dest="disease_level", nargs='?', default="Lowest", help="which disease level would you like to use?", choices=["Highest", "Lowest"])
    parser.add_argument("-s", dest="Save", nargs='?', default=False, help="location of file")
    parser.add_argument("-d", dest="Show", nargs='?', default=True, help="Do you want to show the plot?")

    args = parser.parse_args()
    start = time.time()
    main_summary(data=False, Path=args.Path, level=args.disease_level, save=args.Save, show=str_to_bool(args.Show))
    end = time.time()
    print('completed in {} seconds'.format(end-start))