import pandas as pd
import numpy as np 
import os
import argparse
import json
import time

# Currently use sys to get other script - in Future use package
import os
import sys
path_main = ("/".join(os.path.realpath(__file__).split("/")[:-2]))
sys.path.append(path_main + '/Classes/')
sys.path.append(path_main + '/Utils/')
from media_class import Medium, Supplement, GrowthMedium, Medium_one_hot, Supplement_one_hot, GrowthMedium_one_hot
from help_functions import mean, str_to_bool, str_none_check

def maf_read(data, Path=False):
    '''

    '''
    if Path != False:
        if not Path.endswith(".pkl"):
            Path = Path + "maf_extract_summary.pkl"
        
        data = pd.read_pickle(Path)

    return data

def cnv_read(data, Path=False):
    '''

    '''
    if Path != False:
        if not Path.endswith(".pkl"):
            Path = Path + "cnv_extract.pkl"
        
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

def additional_merge(row, include_disease=None, include_tissue=False, include_dimension=False):
    '''

    '''
    hot_info = one_hot(np.empty(0), np.empty(0))
    if include_disease != None:
        if include_disease == "Highest level":
            disease_onehot = one_hot(np.zeros(len(cache_diseases_highest)), np.array(cache_diseases_highest))
            try:
                inde = cache_diseases_highest.index(str(row["Disease_highest_level"]))
            except:
                inde = cache_diseases_highest.index("Unknown")

            disease_onehot.counts[inde] = 1
            
            hot_info = hot_info.add(disease_onehot)
        elif include_disease == "Lowest level":
            disease_onehot = one_hot(np.zeros(len(cache_diseases_lowest)), np.array(cache_diseases_lowest))
            try:
                inde = cache_diseases_lowest.index(str(row["Disease_lowest_level"]))
            except:
                inde = cache_diseases_lowest.index("Unknown")

            disease_onehot.counts[inde] = 1
            
            hot_info = hot_info.add(disease_onehot)
        else:
            raise KeyError(f"Unrecognized include disease info: {include_disease}. Please use either None, Highest level or Lowest level")
    
    if include_tissue == True:
        tissue_onehot = one_hot(np.zeros(len(cache_tumor_site)), np.array(cache_tumor_site))
        try:
            inde = cache_tumor_site.index(str(row["Tissue_Site"]))
        except:
            inde = cache_tumor_site.index("Unknown")

        tissue_onehot.counts[inde] = 1
        
        hot_info = hot_info.add(tissue_onehot)
    
    if include_dimension == True:
        dimension_onehot = one_hot(np.zeros(len(cache_dimension)), np.array(cache_dimension))
        try:
            inde = cache_dimension.index(str(row["Dimension"]))
        except:
            inde = cache_dimension.index("Unknown")

        dimension_onehot.counts[inde] = 1
        
        hot_info = hot_info.add(dimension_onehot)

    hot_info = hot_info.add(row["cnv_one_hot"])
    row["Flat_one_hot"] = hot_info.add(row['maf_one_hot'])
    return row

def main_merge(data, Path=False, include_cnv=True, include_disease=None, include_tissue=False, include_dimension=False, Save=False, Show=False, Save_cache=False):

    if Path != False:
        if not Path.endswith('/'):
            Path = Path + "/"

    maf_data = maf_read(data=data, Path=Path)

    meta_data = other_data(data=data, Path=Path)

    # Create File name -> Should be in maf.py
    meta_data["File"] = [str(x).split("/")[-1].split(".")[0] for x in meta_data["PANEL_oncotated_maf_mutect2"]]

    data = pd.merge(meta_data, maf_data, on="File")

    # Create global caches
    global cache_diseases_highest
    cache_diseases_highest = data["Disease_highest_level"].unique().tolist()
    global cache_diseases_lowest
    cache_diseases_lowest = data["Disease_lowest_level"].unique().tolist()
    global cache_dimension
    cache_dimension = data["Dimension"].unique().tolist()
    cache_dimension = ["Unknown" if str(x)=="nan" or str(x) == "2D,Suspension" or str(x) == "2D,3D" else str(x) for x in cache_dimension]
    cache_dimension = list(set(cache_dimension))
    global cache_tumor_site
    cache_tumor_site = data["Tissue_Site"].unique().tolist()
    cache_tumor_site = ["Unknown" if str(x)=="nan" else str(x) for x in cache_tumor_site]
    cache_tumor_site = list(set(cache_tumor_site))

    if include_cnv == True:
        cnv_data = cnv_read(data=data, Path=Path)
        data["File"] = [str(x).split(".")[0] for x in data["File"]]
        data = pd.merge(data, cnv_data, on="File")
        data.rename(columns = {'Flat_one_hot': 'maf_one_hot', 'Alt_one_hot': 'maf_Alt_one_hot', 'Alt_2D': 'maf_Alt_2D'}, inplace = True)
        data = data.apply(additional_merge, include_disease=include_disease, include_tissue=include_tissue, include_dimension=include_tissue, axis=1)

    print(data)
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
            Save += "result.json"
            
        with open(Save, 'w') as fp:
            json.dump(data_terra, fp)

    if Save_cache != False:
        # Cache_features
        
        cache_features = data.iloc[0]["Flat_one_hot"]
        tmp_counts = cache_features.counts
        tmp_counts[tmp_counts > 0] = 0
        cache_features.counts = tmp_counts

        # Cache_media
        cache_media = data.iloc[0]["one-hot"]
        tmp_counts = cache_media.media.counts
        tmp_counts[tmp_counts > 0] = 0
        cache_media.media.counts = tmp_counts
        
        tmp_counts = cache_media.supplements.counts
        tmp_counts[tmp_counts > 0] = 0
        cache_media.supplements.counts = tmp_counts

        cache_data = {
            "cache_diseases_highest": cache_diseases_highest,
            "cache_tumor_site": cache_tumor_site,
            "cache_dimension": cache_dimension,
            "cache_media":cache_media.media.levels,
            "cache_supplements":cache_media.supplements.levels,
            "cache_all": cache_features.levels.tolist()
        }

        if not Save_cache.endswith("/"):
            Save_cache += "/"

        with open(Save_cache+"cache_features.json", 'w') as fp:
            json.dump(cache_data, fp)

        cache_features = pd.DataFrame({"Features":cache_features, "Media":cache_media}, index=[0])
        cache_features.to_pickle(Save_cache+"cache_features.pkl", protocol=4)

    if Show == True:
        print(data)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Combine panel data with main frame")
    parser.add_argument("Path", help="path to directory with all files")
    parser.add_argument("-c", dest="CNV", nargs='?', default=False, help="Do you want to include copy number variations?")
    parser.add_argument("-n", dest="DiseaseName", nargs='?', default=None, help="Do you want to include disease name?", choices=['Highest level', 'Lowest level', 'None'])
    parser.add_argument("-t", dest="Tissue", nargs='?', default=False, help="Do you want to include tissue site?")
    parser.add_argument("-r", dest="Dimension", nargs='?', default=False, help="Do you want to include dimension?")
    parser.add_argument("-s", dest="Save", nargs='?', default=False, help="location of file")
    parser.add_argument("-sc", dest="Save_cache", nargs='?', default=False, help="location to store cache")
    parser.add_argument("-d", dest="Show", nargs='?', default=True, help="Do you want to show the plot?")

    args = parser.parse_args()
    start = time.time()
    main_merge(data=False, Path=args.Path, include_cnv=str_to_bool(args.CNV), include_disease=str_none_check(args.DiseaseName), include_tissue=str_to_bool(args.Tissue), include_dimension=str_to_bool(args.Dimension), Save=args.Save, Show=str_to_bool(args.Show), Save_cache=args.Save_cache)
    end = time.time()
    print('completed in {} seconds'.format(end-start))