### Author Douwe Spaanderman - 3 June 2020 ###

# This script creates a heatmap of the media type against the disease type names
# Note that currently disease types are still weird

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
from sklearn.cluster import AgglomerativeClustering
import argparse

# Currently use sys to get other script
import sys
sys.path.insert(1, "../Classes/")
from media_class import Medium, Supplement, GrowthMedium, Medium_one_hot, Supplement_one_hot, GrowthMedium_one_hot

def extract_data(data):
    '''
    Changing data for plotting
    
    input:
    data = is a loaded pandas dataframe
                
    return:
    heatmap_media, heatmap_supplements = dataframe with media/supplement 
                                            occurences
    '''
    #Create an empty numpy array for the heatmap in which to count occurences
    # First get all disease types and all media/supplements
    disease_types = list(set(data["Primary_Disease"]))
    disease_types = sorted([x for x in disease_types if not str(x) == "nan"])
    
    all_media = sorted(list(set().union(*(d.keys() for d in [x.media for x in data["media_class"]]))))
    all_suple = sorted(list(set().union(*(d.keys() for d in [x.supplements for x in data["media_class"]]))))
    
    media_array = np.zeros([len(disease_types), len(all_media)])
    suple_array = np.zeros([len(disease_types), len(all_suple)])
    
    # Loop for each disease:
    for i, disease in enumerate(disease_types):
        tmp_data = data[data["Primary_Disease"] == disease]
        if not tmp_data.empty:
            #Get all keys
            media = [list(x.media.keys()) for x in tmp_data["media_class"]]
            suple = [list(x.supplements.keys()) for x in tmp_data["media_class"]]
                
            #Flatten list
            media = [item for sublist in media for item in sublist]
            suple = [item for sublist in suple for item in sublist]
                
            #Count occurences in list
            media_counter = dict(Counter(media))
            suple_counter = dict(Counter(suple))
                
            #Get index in array
            index_media = [i for i, item in enumerate(all_media) if item in list(media_counter.keys())]
            index_suple = [i for i, item in enumerate(all_suple) if item in list(suple_counter.keys())]
                
            #Change 0 to occurences
            np.put(media_array[i], index_media, list(media_counter.values()))
            np.put(suple_array[i], index_suple, list(suple_counter.values()))  
            
    heatmap_media = pd.DataFrame(data=media_array, index=disease_types, columns=all_media)
    heatmap_supplements = pd.DataFrame(data=suple_array, index=disease_types, columns=all_suple)
    
    return heatmap_media, heatmap_supplements

def Cluster_order(heatmap_data):
    '''
    hierachrical clustering of heatmap data to find some structure (more readable)

    input:
    heatmap_data = data from extract_data

    output:
    reindexed version of heatmap data
    '''
    # Transpose to cluster on media
    heatmap_data = heatmap_data.T

    #Clustering with distance - and n_clusters at 0, will create an individual group for all
    clustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit_predict(heatmap_data.values)
    
    #index based on clusters and transpose back
    heatmap_data = heatmap_data.iloc[list(clustering)].T

    return heatmap_data

def Count_order(heatmap_data):
    '''
    Counting max columns in heatmap data to find some structure (more readable)

    input:
    heatmap_data = data from extract_data

    output:
    reindexed version of heatmap data
    '''
    # Count max values for media and use ascending sort
    heatmap_data = heatmap_data.loc[:, heatmap_data.sum().sort_values(ascending=False).index]

    return heatmap_data

def plotting(heatmap_media, heatmap_supplements, Scale="log2", save=False, show=True, extention=""):
    '''
    Heatmap plotting of both heatmap_media and heatmap_supplements (would be nicer if it would be one function)

    input:
    heatmap_media = input data from extract_data for media
    heatmap_supplements = input data from extract_data for supplements
    Scale = how to scale the heatmap values=["log2", "normalized", "log10"]
    save = if you want to save the plot (default is True)
    show = if you want to show the plot (default is True)
    SavePath = If you want to save the plot where to save it (default is in Data/Ongoing folder)
    '''
    if Scale == "log2":
        heatmap_media = np.log2(heatmap_media + 1)
        heatmap_supplements = np.log2(heatmap_supplements + 1)
    elif Scale == "log10":
        heatmap_media = np.log10(heatmap_media + 1)
        heatmap_supplements = np.log10(heatmap_supplements + 1)
    elif Scale == "normalized":
        heatmap_media = heatmap_media.div(heatmap_media.sum(axis=1), axis=0)
        heatmap_supplements = heatmap_supplements.div(heatmap_supplements.sum(axis=1), axis=0)
    else:
        warnings.warn(f'unrecognized scale was provided for plotting: {Scale}, therefore data has not been changed')

    width_ratio = len(heatmap_media.columns)/len(heatmap_supplements.columns)

    # Get a combined legend using the max value
    max_value = np.max([np.max(heatmap_media.values), np.max(heatmap_supplements.values)])

    f, ax = plt.subplots(1, 2, figsize=(18, 18*width_ratio), gridspec_kw={'width_ratios': [width_ratio, 1]})
    media_plot = sns.heatmap(heatmap_media, ax=ax[0], square=False, vmin=0, vmax=max_value, cbar=False)
    suple_plot = sns.heatmap(heatmap_supplements, ax=ax[1], square=False, vmin=0, vmax=max_value, yticklabels=False)

    media_plot.set_xticklabels(
        list(heatmap_media.columns), 
        horizontalalignment='center',
        fontsize=9,
    )

    media_plot.set_title(
        label='Media:\n',
        loc="left",
        fontsize=20, 
    )

    suple_plot.set_xticklabels(
        list(heatmap_supplements.columns), 
        horizontalalignment='center',
        fontsize=9,
    )

    suple_plot.set_title(
        label='Supplements:\n',
        loc="left",
        fontsize=20, 
    )

    f.tight_layout()

    if save != False:
        plt.savefig(save + extention + "heatmap_media_matrix.png")
    if show == True:
        plt.show()
    if show == False and save == False:
        warnings.warn('you are not checking the input data for media/supplements')

def heatmap_media_matrix(data, Path=False, Order="Occurence", Scale="log2", save=False, show=True):
    '''
    Main script for plotting heatmap media matrix

    input:
    data = loaded file after media.py
    Path = location of file after media.py
    Order = how to order the heatmap choices=["Occurence", "Clustered", "Nothing", "All"]
    Scale = how to scale the heatmap values=["log2", "normalized", "log10"]
    save = if you want to save the plot (default is True)
    show = if you want to show the plot (default is True)
    SavePath = If you want to save the plot where to save it (default is in Data/Ongoing folder)
    '''
    # Read file if using command line
    if Path != False:
        if not Path.endswith(".pkl"):
            Path = Path + "after_media.pkl"
        
        data = pd.read_pickle(Path)
    
    media_data, supplement_data = extract_data(data)

    if Order == "Clustered":
        media_data = Cluster_order(media_data)
        supplement_data = Cluster_order(supplement_data)
        plotting(media_data, supplement_data, save=save, show=show, extention="clustering_")
    elif Order == "Occurence":
        media_data = Count_order(media_data)
        supplement_data = Count_order(supplement_data)
        plotting(media_data, supplement_data, save=save, show=show, extention="occurence_")
    elif Order == "Nothing":
        plotting(media_data, supplement_data, save=save, show=show)
    elif Order == "All":
        # Nothing
        plotting(media_data, supplement_data, save=save, show=show)

        #Count occurrences
        tmp_media_data = Count_order(media_data)
        tmp_supplement_data = Count_order(supplement_data)
        plotting(tmp_media_data, tmp_supplement_data, save=save, show=show, extention="occurence_")

        #Clustered occurrences
        tmp_media_data = Cluster_order(media_data)
        tmp_supplement_data = Cluster_order(supplement_data)
        plotting(tmp_media_data, tmp_supplement_data, save=save, show=show, extention="clustering_")
    else:
        raise KeyError(f"Wrong Order was provided: {Order}, please pick either Occurence, Clustered, Nothing, All.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Change media encoding")
    parser.add_argument("Path", help="path to terra workspace filtered file after media.py")
    parser.add_argument("-s", dest="Save", nargs='?', default=False, help="location of file")
    parser.add_argument("-d", dest="Show", nargs='?', default=True, help="Do you want to show the plot?")
    parser.add_argument("-o", dest="Data_ordering", nargs='?', default="Occurence", help="would you like to do order the data", choices=["Occurence", "Clustered", "Nothing", "All"])
    parser.add_argument("-c", dest="Data_scaling", nargs='?', default="log2", help="how would you like to scale the data", choices=["log2", "normalized", "log10"])

    args = parser.parse_args()
    start = time.time()
    heatmap_media_matrix(data=False, Path=args.Path, Order=args.Data_ordering, save=args.Save, show=bool(args.Show), Scale=args.Data_scaling)
    end = time.time()
    print('completed in {} seconds'.format(end-start))