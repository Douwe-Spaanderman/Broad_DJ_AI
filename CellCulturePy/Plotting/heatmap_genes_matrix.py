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

def extract_data(meta_data, maf_data, level="Lowest"):
    '''
    Changing data for plotting
    
    input:
    meta_data = is a loaded pandas dataframe after media.py
    maf_data = is a loaded pandas dataframe after maf.py
                
    return:
    heatmap_data = dataframe with gene occurence
    counts_symbols = counter of symbols
    counts_disease = counter of disease
    '''
    meta_data["file"] = [str(x).split("/")[-1].split(".")[0] for x in meta_data["PANEL_oncotated_maf_mutect2"]]

    if level == "Highest":
        meta_data = meta_data[meta_data["Disease_highest_level"] != "Unknown"]
        disease_types = list(set(meta_data["Disease_highest_level"]))
        
    elif level == "Lowest":
        meta_data = meta_data[meta_data["Disease_lowest_level"] != "Unknown"]
        disease_types = list(set(meta_data["Disease_lowest_level"]))
    else:
        raise KeyError(f"Wierd disease level was passed: {level}. please provide either Highest or Lowest")

    # Counter symbols
    counts_symbols = Counter(maf_data["Hugo_Symbol"])
    counts_symbols = dict(counts_symbols.most_common(100))

    maf_data = maf_data[maf_data["Hugo_Symbol"].isin(list(counts_symbols.keys()))]

    counts_symbols = pd.Series(counts_symbols, name='Count')
    counts_symbols.index.name = 'Hugo_Symbol'
    counts_symbols = counts_symbols.reset_index()

    # Counter disease
    counts_disease = Counter(meta_data["Disease_lowest_level"])
    counts_disease = dict(counts_disease.most_common(len(counts_disease)))
    counts_disease = pd.Series(counts_disease, name='Count')
    counts_disease.index.name = 'Disease_lowest_level'
    counts_disease = counts_disease.reset_index()

    # Merge and transform into matrix 
    heatmap_data = pd.merge(maf_data, meta_data, on="file")

    if level == "Highest":
        heatmap_data = heatmap_data.groupby(["Disease_highest_level", "Hugo_Symbol"]).size().reset_index(name="Count")
        heatmap_data = heatmap_data.pivot(index='Hugo_Symbol', columns='Disease_highest_level', values='Count')
        heatmap_data.index = heatmap_data.index.str.strip()
        heatmap_data = heatmap_data.reindex(counts_symbols["Hugo_Symbol"].tolist())
        heatmap_data = heatmap_data.reindex(counts_disease["Disease_highest_level"].tolist(), axis=1)
        heatmap_data = heatmap_data.fillna(0)
        
    elif level == "Lowest":
        heatmap_data = heatmap_data.groupby(["Disease_lowest_level", "Hugo_Symbol"]).size().reset_index(name="Count")
        heatmap_data = heatmap_data.pivot(index='Hugo_Symbol', columns='Disease_lowest_level', values='Count')
        heatmap_data.index = heatmap_data.index.str.strip()
        heatmap_data = heatmap_data.reindex(counts_symbols["Hugo_Symbol"].tolist())
        heatmap_data = heatmap_data.reindex(counts_disease["Disease_lowest_level"].tolist(), axis=1)
        heatmap_data = heatmap_data.fillna(0)
    else:
        raise KeyError(f"Wierd disease level was passed: {level}. please provide either Highest or Lowest")
    
    return heatmap_data, counts_symbols, counts_disease

def plotting_normal(plot_data, counts_symbols, counts_disease, save=False, show=True, extention=""):
    '''
    Heatmap plotting genes

    input:
    plot_data = input data from extract_data
    counts_symbols = counter of symbols
    counts_disease = counter of disease
    save = if you want to save the plot (default is True)
    show = if you want to show the plot (default is True)
    SavePath = If you want to save the plot where to save it (default is in Data/Ongoing folder)
    '''

    width_ratio = len(plot_data.columns)

    # Get a combined legend using the max value
    max_value = np.max(np.max(plot_data))

    f, ax = plt.subplots(2, 2, figsize=(30, 45), gridspec_kw={'width_ratios': [1, 5], 'height_ratios': [1, 7.5]})
    ax_genes = ax[1][0].plot(counts_symbols["Count"], counts_symbols["Hugo_Symbol"])
    ax_disease = ax[0][1].plot(counts_disease["Disease_lowest_level"], counts_disease["Count"])
    ax_main = sns.heatmap(plot_data, ax=ax[1][1], square=False, vmin=0, cbar=False, cbar_kws={"shrink": .60})

    ax[0][1].set_xlim(xmin=0, xmax=len(counts_disease)-1)
    ax[1][0].set_ylim(ymin=0, ymax=len(counts_symbols)-1)

    ax[0][1].set_xticks([])
    ax[1][0].set_yticks([])
    ax[1][0].invert_yaxis()
    ax[1][0].xaxis.tick_top()
    ax[1][0].invert_xaxis()

    ax[1][1].yaxis.set_label_text('')
    ax[1][1].xaxis.set_label_text('')
    ax[1][1].yaxis.tick_right()

    ax[1][1].set_yticklabels(counts_symbols["Hugo_Symbol"], rotation=0, ha='left', fontsize=16)
    ax[1][1].set_xticklabels(counts_disease["Disease_lowest_level"], rotation=90, ha='right', fontsize=16)

    ax[0][1].set_yticklabels([0,0,100,200,300,400,500,600], rotation=0, fontsize=16)
    ax[1][0].set_xticklabels([0,100,200,300,400], rotation=0, fontsize=16)

    f.delaxes(ax[0][0])
    

    f.tight_layout()
    ax[1][1].set_xticklabels(counts_disease["Disease_lowest_level"], rotation=45, ha='right', fontsize=16)

    if save != False:
        if not save.endswith("Figures/"):
            save += "Figures/"
        f.savefig(save + extention + "heatmap_genes_matrix.png")
    if show == True:
        f.show()
    if show == False and save == False:
        warnings.warn('you are not checking the input data for media/supplements')

def plotting_cluster(plot_data, save=False, show=True, extention=""):
    '''
    Heatmap plotting genes

    input:
    plot_data = input data from extract_data
    save = if you want to save the plot (default is True)
    show = if you want to show the plot (default is True)
    SavePath = If you want to save the plot where to save it (default is in Data/Ongoing folder)
    '''
    
    ax = sns.clustermap(plot_main, figsize=(30, 45))
    
    plt.setp(ax.ax_heatmap.get_yticklabels(), rotation=0, ha='left', fontsize=16)  # For y axis
    plt.setp(ax.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=16) # For x axis

    ax.ax_heatmap.set_xlabel("")
    ax.ax_heatmap.set_ylabel("")

    if save != False:
        if not save.endswith("Figures/"):
            save += "Figures/"
        plt.savefig(save + extention + "heatmap_genes_matrix.png")
    if show == True:
        plt.show()
    if show == False and save == False:
        warnings.warn('you are not checking the input data for media/supplements')

def heatmap_genes_matrix(meta_data, maf_data, Path_meta=False, Path_maf=False, Order="Nothing", Scale="Normalized", level="Lowest", save=False, show=True):
    '''
    Main script for plotting heatmap genes matrix

    input:
    data_meta = loaded file after media.py
    data_maf = loaded file after maf.py
    Path_meta = location of file after media.py
    Path_maf = location of file after maf.py
    Order = how to order the heatmap choices=["Clustered", "Nothing", "Both"]
    Scale = how to scale the heatmap values=["log2", "Normalized", "log10"]
    save = if you want to save the plot (default is True)
    show = if you want to show the plot (default is True)
    SavePath = If you want to save the plot where to save it (default is in Data/Ongoing folder)
    '''
    # Read file if using command line
    if Path_meta != False:
        if not Path_meta.endswith(".pkl"):
            Path_meta = Path_meta + "after_media.pkl"
        
        meta_data = pd.read_pickle(Path_meta)

    if Path_maf != False:
        if not Path_maf.endswith(".pkl"):
            Path_maf = Path_maf + "maf_extract.pkl"
        
        maf_data = pd.read_pickle(Path_maf)
    
    heatmap_data, counts_symbols, counts_disease = extract_data(meta_data, maf_data, level=level)

    # Scaling
    if Scale == "log2":
        heatmap_data = np.log2(heatmap_data + 1)
    elif Scale == "log10":
        heatmap_data = np.log10(heatmap_data + 1)
    elif Scale == "Normalized":
        heatmap_data = heatmap_data / counts_disease["Count"].tolist()
        heatmap_data = heatmap_data.apply(lambda x: np.where((x > 1), 1, x))
    else:
        warnings.warn(f'unrecognized scale was provided for plotting: {Scale}, therefore data has not been changed')

    # Ordering
    if Order == "Clustered":
        plotting_cluster(heatmap_data, save=save, show=show, extention="clustering_")
    elif Order == "Nothing":
        plotting_normal(heatmap_data, counts_symbols, counts_disease, save=save, show=show)
    elif Order == "Both":
        # Nothing
        plotting_normal(heatmap_data, counts_symbols, counts_disease, save=save, show=show)

        #Clustered occurrences
        heatmap_data = Cluster_order(heatmap_data, counts_symbols, counts_disease)
        plotting_cluster(heatmap_data, save=save, show=show, extention="clustering_")
    else:
        raise KeyError(f"Wrong Order was provided: {Order}, please pick either Clustered, Nothing, Both.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Heatmap matrix of genes in samples")
    parser.add_argument("Path", help="path to terra workspace filtered file after media.py")
    parser.add_argument("Maf", help="path to maf_extract.pkl file after maf.py")
    parser.add_argument("-s", dest="Save", nargs='?', default=False, help="location of file")
    parser.add_argument("-d", dest="Show", nargs='?', default=True, help="Do you want to show the plot?")
    parser.add_argument("-l", dest="disease_level", nargs='?', default="Lowest", help="which disease level would you like to use?", choices=["Highest", "Lowest"])
    parser.add_argument("-o", dest="Data_ordering", nargs='?', default="Nothing", help="would you like to do order the data", choices=["Clustered", "Nothing", "Both"])
    parser.add_argument("-c", dest="Data_scaling", nargs='?', default="Normalized", help="how would you like to scale the data", choices=["log2", "Normalized", "log10"])

    args = parser.parse_args()
    start = time.time()
    heatmap_genes_matrix(meta_data=False, maf_data=False, Path_meta=args.Path, Path_maf=args.Maf, Order=args.Data_ordering, level=args.disease_level, save=args.Save, show=str_to_bool(args.Show), Scale=args.Data_scaling)
    end = time.time()
    print('completed in {} seconds'.format(end-start))