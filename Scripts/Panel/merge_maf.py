import pandas as pd
import numpy as np 
import os

# Currently use sys to get other script
import sys
sys.path.insert(1, "../Classes/")
from gene_one_hot import one_hot
def maf_data(data, Path):
    '''

    '''
    if Path != False:
        if not Path.endswith(".pkl"):
            Path = Path + "maf_extract_summary.pkl"
        
        data = pd.read_pickle(Path)

    return data

maf_data = maf_data(data=False, Path="../Data/Ongoing/")

# Currently use sys to get other script
import sys
sys.path.insert(1, "../Classes/")
from media_class import Medium, Supplement, GrowthMedium, Medium_one_hot, Supplement_one_hot, GrowthMedium_one_hot

def other_data(data, Path):
    '''

    '''
    if Path != False:
        if not Path.endswith(".pkl"):
            Path = Path + "after_media.pkl"
        
        data = pd.read_pickle(Path)

    return data 

meta_data = other_data(data=False, Path="../Data/Ongoing/after_media.pkl")

# Create File name -> Should be in maf.py
meta_data["File"] = [str(x).split("/")[-1] for x in meta_data["PANEL_oncotated_maf_mutect2"]]

data = pd.merge(meta_data, maf_data, on="File")

# Input with media as input
input_data = data.apply(lambda x: np.append(x["Flat_one_hot"].counts, x["one-hot"].media.counts), axis=1)
input_data = np.vstack(input_data)

#Specify Ouput 
output_data = np.vstack(data["Growth"])
output_data = np.ravel(output_data)

# With media as output next to growing
#Specify input
input_data_neural = np.vstack([x.counts for x in data["Flat_one_hot"]])

#Specify Ouput 
# Add array which has one-hot media with growing not growing
output_data_neural = data.apply(lambda x: np.append(x["one-hot"].media.counts, x["Growth"]), axis=1)
output_data_neural = np.vstack(output_data_neural)

data_terra = {"input_data": input_data.tolist(), "output_data": output_data.tolist(), "input_data_neural": input_data_neural.tolist(), "output_data_neural":output_data_neural.tolist()}

import json
with open('../Data/Ongoing/result.json', 'w') as fp:
    json.dump(data_terra, fp)
