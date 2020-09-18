import pandas as pd
import numpy as np 
import os
import argparse
import json
import time

# Currently use sys to get other script
import sys
sys.path.insert(1, "/Users/dspaande/Documents/GitProjects/Broad_DJ_AI/DeepGrowth/Classes/")
from media_class import Medium, Supplement, GrowthMedium, Medium_one_hot, Supplement_one_hot, GrowthMedium_one_hot
from gene_one_hot import one_hot

# Currently use sys to get other script
sys.path.insert(1, "/Users/dspaande/Documents/GitProjects/Broad_DJ_AI/DeepGrowth/Utils/")
from help_functions import mean, str_to_bool

def maf_read(data, Path=False):
    '''

    '''
    if Path != False:
        if not Path.endswith(".pkl"):
            Path = Path + "maf_extract_summary.pkl"
        
        data = pd.read_pickle(Path)

    return data

def other_data(data, Path=False):
    '''

    '''
    if Path != False:
        if not Path.endswith(".pkl"):
            Path = Path + "after_media.pkl"
        
        data = pd.read_pickle(Path)

    return data 

def main_merge(data, Path=False, Save=False, Show=False):

    if Path != False:
        if not Path.endswith('/'):
            Path = Path + "/"

    maf_data = maf_read(data=data, Path=Path)

    meta_data = other_data(data=data, Path=Path)

    # Create File name -> Should be in maf.py
    meta_data["File"] = [str(x).split("/")[-1] for x in meta_data["PANEL_oncotated_maf_mutect2"]]

    data = pd.merge(meta_data, maf_data, on="File")

    # Input with media as input
    input_data = data.apply(lambda x: np.append(x["Flat_one_hot"].counts, x["one-hot"].media.counts), axis=1)
    input_data = np.vstack(input_data)

    #Specify Ouput 
    output_data = np.vstack(data["Growth"])
    output_data = np.ravel(output_data)

    # Input with media+supplements as input
    input_data_supplements = data.apply(lambda x: np.append(x["Flat_one_hot"].counts, np.append(x["one-hot"].media.counts, x["one-hot"].supplements.counts)), axis=1)
    input_data_supplements = np.vstack(input_data_supplements)

    #Specify Ouput 
    output_data_supplements = np.vstack(data["Growth"])
    output_data_supplements = np.ravel(output_data_supplements)

    # With media as output next to growing
    #Specify input
    input_data_neural = np.vstack([x.counts for x in data["Flat_one_hot"]])

    #Specify Ouput 
    # Add array which has one-hot media with growing not growing
    output_data_neural = data.apply(lambda x: np.append(x["one-hot"].media.counts, x["Growth"]), axis=1)
    output_data_neural = np.vstack(output_data_neural)

    data_terra = {"input_data": input_data.tolist(), 
              "output_data": output_data.tolist(), 
              "input_data_supplements":input_data_supplements.tolist(), 
              "output_data_supplements": output_data_supplements.tolist(), 
              "input_data_neural": input_data_neural.tolist(), 
              "output_data_neural":output_data_neural.tolist()
             }
             
    if Save != False:
        if not Save.endswith(".json"):
            Save = Save + "result.json"
            
        with open(Save, 'w') as fp:
            json.dump(data_terra, fp)

    if Show == True:
        print(data)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Combine panel data with main frame")
    parser.add_argument("Path", help="path to directory with all files")
    parser.add_argument("-s", dest="Save", nargs='?', default=False, help="location of file")
    parser.add_argument("-d", dest="Show", nargs='?', default=True, help="Do you want to show the plot?")

    args = parser.parse_args()
    start = time.time()
    main_merge(data=False, Path=args.Path, Save=args.Save, Show=str_to_bool(args.Show))
    end = time.time()
    print('completed in {} seconds'.format(end-start))