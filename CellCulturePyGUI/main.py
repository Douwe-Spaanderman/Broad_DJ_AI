### Author Douwe Spaanderman - 29 September 2020 ###
import pandas as pd
import numpy as np
import json
from gooey import Gooey, GooeyParser
import _pickle as cPickle

from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

import sys
import os
path_main = "/".join(os.path.realpath(__file__).split("/")[:-1])
print(path_main)
sys.path.append(path_main + '/Classes/')
sys.path.append(path_main + '/Utils/')
from media_class import Medium, Supplement, GrowthMedium, Medium_one_hot, Supplement_one_hot, GrowthMedium_one_hot
from gene_one_hot import one_hot
from help_functions import mean, str_to_bool, str_none_check
from message import display_message

@Gooey(dump_build_config=True, program_name="CellCulturePy")
def main():
    '''

    '''
    Cache_pickle = pd.read_pickle('Data/cache_features.pkl')

    with open("Data/cache_features.json", "r") as read_file:
        Cache_json = json.load(read_file)

    parser = GooeyParser(description="Predicting media conditions with genomic profile")
    parser.add_argument("TumorType", help="what is your tumor type", choices=Cache_json["cache_diseases_highest"])
    parser.add_argument("Tissue", help="what is your tissue type", choices=Cache_json["cache_tumor_site"])
    parser.add_argument("Dimension", help="what is your growing dimension", choices=Cache_json["cache_dimension"])
    parser.add_argument("maf", help="Select maf file from TWIST (mutect1 or 2)", widget="FileChooser")
    parser.add_argument("cnv", help="Select cnv file from TWIST (.tumor.called)", widget="FileChooser")
    parser.add_argument("-m", dest="Media", nargs='+', default=None, choices=Cache_json["cache_media"], help="you can select one or multiple media types you want to look for", widget="Listbox")
    parser.add_argument("-s", dest="Supplements", action="store_true", help="Do you want to include looking for supplements (default: No)")

    args = parser.parse_args()
    predicted = predict(Cache_json=Cache_json, Cache_pickle=Cache_pickle, TumorType=args.TumorType, Tissue=args.Tissue, Dimension=args.Dimension, maf=args.maf, cnv=args.cnv, media=False, supplements=False)
    display_message()
    
def maf_extract(maf):
    '''

    '''
    id_ = []
    data_dict= {}
    file_name = maf.split('/')[-1]
    i = 0
    with open(maf, 'r', encoding="latin-1") as f:
        try:
            for line in f:
                if line.startswith("#"):
                    continue
                elif not id_:
                    id_ = line.replace('\n', '').split('\t')
                else:
                    data_dict[i] = line.replace('\n', '').split('\t')
                    i += 1
        except:
            warnings.warn(f"File: {file_name}, had problems with unrecognizable symbols", DeprecationWarning)

    maf_frame = pd.DataFrame.from_dict(data_dict, orient="index", columns=id_)

    maf_frame = maf_frame[~maf_frame["Variant_Classification"].isin(["Intron", "lincRNA", "IGR", "5'Flank", "5'UTR", "Silent", "3'UTR", "RNA"])]
    
    return maf_frame, file_name

def cnv_extract(cnv):
    '''

    '''
    file_name = cnv.split('/')[-1]
    cnv_frame = pd.read_csv(cnv, sep="\t")

    chromosomelist = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13",   "14", "15", "16", "17", "18", "19", "20", "21", "22", "X", "Y"]                         
                                                                                            
    cnv_data = {}                                                                               
    for chromosoom in chromosomelist:
        cnv_tmp = cnv_frame[cnv_frame["Chromosome"] == chromosoom]
        if type(cnv_tmp) == pd.core.series.Series or cnv_tmp.empty == True:
            cnv_data[chromosoom] = 1
        elif len(cnv_tmp) > 1:
            cnv_data[chromosoom] = (sum(cnv_tmp["Num_Probes"] * cnv_tmp["Segment_Mean"]) / sum(cnv_tmp["Num_Probes"]))
        else:
            cnv_data[chromosoom] = cnv_tmp["Segment_Mean"].tolist()[0]

    cnv_data = pd.Series(cnv_data, name='Value')
    cnv_data.index.name = 'Chromosome'
    cnv_data = cnv_data.reset_index()
    return cnv_data, file_name

def predict(Cache_json, Cache_pickle, TumorType, Tissue, Dimension, maf, cnv, media=False, supplements=False):
    '''

    '''
    one_hot_features = Cache_pickle.iloc[0]["Features"]
    one_hot_media = Cache_pickle.iloc[0]["Media"]

    #TumorType info
    idx = list(one_hot_features.levels).index(TumorType)
    counts = one_hot_features.counts
    counts[idx] = 1
    one_hot_features.counts = counts

    #Tissue info
    levels = list(one_hot_features.levels)
    idx = [i for i in range(len(levels)) if levels[i] == Tissue]
    if len(idx) > 1:
        idx = idx[1]
    else:
        idx = idx[0]
    counts = one_hot_features.counts
    counts[idx] = 1
    one_hot_features.counts = counts

    #Dimension info
    levels = list(one_hot_features.levels)
    idx = [i for i in range(len(levels)) if levels[i] == Dimension]
    if len(idx) > 1:
        idx = idx[1]
    else:
        idx = idx[0]
    counts = one_hot_features.counts
    counts[idx] = 1
    one_hot_features.counts = counts

    # Maf info
    maf_data, maf_name = maf_extract(maf)

    if maf_data.empty:
        print("Mutation file is empty as nothing was probably protein coding")
    else:
        for i, row in maf_data.iterrows():
            levels = list(one_hot_features.levels)
            idx = [i for i in range(len(levels)) if levels[i] == row["Hugo_Symbol"]]
            if not idx:
                continue
            else:
                counts = one_hot_features.counts
                counts[idx] = 1
                one_hot_features.counts = counts

    # CNV Info
    cnv_data, cnv_name = cnv_extract(cnv)
    if cnv_data.empty:
        print("CNV file is empty as no copy number variations")
    else:
        for i, row in cnv_data.iterrows():
            idx = list(one_hot_features.levels).index(row["Chromosome"])
            counts = one_hot_features.counts
            counts[idx] = row["Value"]
            one_hot_features.counts = counts

    #Load model
    with open('Models/RF_model.sav', 'rb') as f:
        rf = cPickle.load(f)

    #Loop through media append to features and
    media = one_hot_media.media
    length_loop = len(media.counts)
    input_data = []
    media_name = []
    for i in range(0, length_loop):
        z = True
        while z == True:
            y = range(i, length_loop)
            for j in y:
                media_tmp = media.counts
                if i == j:
                    media_tmp[i] = 1
                    media_name.append(f"{media.levels[i]} 100%")
                else:
                    media_tmp[i] = 1
                    media_tmp[j] = 1
                    media_name.append(f"{media.levels[i]} 50% and {media.levels[j]} 50%")

                one_hot = np.append(one_hot_features.counts, media_tmp)
                input_data.append(one_hot)

                if i == j:
                    media_tmp[i] = 0
                else:
                    media_tmp[i] = 0
                    media_tmp[j] = 0

                if j == length_loop-1:
                    z = False
        else:
            continue

    predictions_proba = rf.predict_proba(input_data)

    # Get predictions for growing
    predictions = [predictions_proba[i,1] for i in range(len(predictions_proba))]

    # Get top 10 values
    idx = np.argsort(-np.array(predictions))[:10]
    top = [predictions[i] for i in idx]
    top_media = [media_name[i] for i in idx]
    print(top_media)

    return predictions, top

if __name__ == '__main__':
    main()