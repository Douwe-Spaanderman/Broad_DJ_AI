### Author Douwe Spaanderman - 29 September 2020 ###
import pandas as pd
import numpy as np
import json
from gooey import Gooey, GooeyParser
import _pickle as cPickle
from collections import Counter
import warnings
import webbrowser
import time

from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

import sys
import os
path_main = "/".join(os.path.realpath(__file__).split("/")[:-1])
sys.path.append(path_main + '/Classes/')
sys.path.append(path_main + '/Utils/')
from media_class import Medium, Supplement, GrowthMedium, Medium_one_hot, Supplement_one_hot, GrowthMedium_one_hot
from gene_one_hot import one_hot
from help_functions import mean, str_to_bool, str_none_check
from message import display_message

@Gooey(dump_build_config=False, 
program_name="CellCulturePy",
richtext_controls=True, 
required_cols=3, 
optional_cols=1,
default_size=(1300, 800))
def main():
    '''

    '''
    Cache_pickle = pd.read_pickle('Data/cache_features.pkl')

    with open("Data/cache_features.json", "r") as read_file:
        Cache_json = json.load(read_file)

    # Assign variables
    TumorType_arg = Cache_json["cache_diseases_highest"]
    TumorType_arg.sort()
    Tissue_arg = Cache_json["cache_tumor_site"]
    Tissue_arg.sort()

    parser = GooeyParser(description="Predicting media conditions with genomic profile")
    parser.add_argument("TumorType", help="what is your tumor type", choices=TumorType_arg)
    parser.add_argument("Tissue", help="what is your tissue type", choices=Tissue_arg)
    parser.add_argument("Dimension", help="what is your growing dimension", choices=Cache_json["cache_dimension"])
    parser.add_argument("maf", help="Select maf file from TWIST (mutect1 or 2)", widget="FileChooser")
    parser.add_argument("cnv", help="Select cnv file from TWIST (.tumor.called)", widget="FileChooser")
    parser.add_argument("-m", dest="Media", nargs='+', default=False, choices=Cache_json["cache_media"], help="you can select one or multiple media types you want to look for", widget="Listbox")
    parser.add_argument("-s", dest="Supplements", action="store_true", default=False, help="Do you want to include looking for supplements (default: No)")

    args = parser.parse_args()
    display_message(part=1)
    predict(Cache_json=Cache_json, Cache_pickle=Cache_pickle, TumorType=args.TumorType, Tissue=args.Tissue, Dimension=args.Dimension, maf=args.maf, cnv=args.cnv, media=str_to_bool(args.Media), supplements=False)
    display_message(part=2)
    

    #Displaying
    path_main = ("/".join(os.path.realpath(__file__).split("/")[:-1]))
    webbrowser.open('file://' + path_main + "/tmp.html")
    time.sleep(5)
    os.remove(path_main + "/tmp.html")

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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rf = cPickle.load(f)

    # Get media input
    if media != False:
        if len(media) == 1:
            print("you only added 1 media type, so will use all")
        else:
            print("Media selection currently not implemented, using all media types")

    #Loop through media append to features
    media = one_hot_media.media
    length_loop = len(media.counts)
    input_data = []
    media_name = []
    for i in range(0, length_loop):
        for j in range(0, length_loop):
            media_tmp = media.counts
            if i == j:
                media_tmp[i] = 1
                media_name.append(f"{media.levels[i]}")
            else:
                media_tmp[i] = 1
                media_tmp[j] = 1
                media_name.append(f"{media.levels[i]} and {media.levels[j]}")

            one_hot = np.append(one_hot_features.counts, media_tmp)
            input_data.append(one_hot)

            if i == j:
                media_tmp[i] = 0
            else:
                media_tmp[i] = 0
                media_tmp[j] = 0
        else:
            continue

    predictions_proba = rf.predict_proba(input_data)

    # Get predictions for growing
    predictions = [predictions_proba[i,1] for i in range(len(predictions_proba))]

    # Get top 10 values
    idx = np.argsort(-np.array(predictions))
    order_pred = [predictions[i] for i in idx]
    order_media = [media_name[i] for i in idx]

    # Make this readable and not overly populated with same media
    pred_df = pd.DataFrame({"Media": order_media, "value": order_pred})
    dis = [x.split(" and ") for x in pred_df["Media"]]
    dis = [x+["NaN"] if len(x) == 1 else x for x in dis]
    pred_df[["Media1","Media2"]] = pd.DataFrame(dis)

    # Create a readable dataframe
    ranked_media = pred_df[pred_df["Media2"] == "NaN"]["Media1"].tolist()
    output_df = []
    for med in ranked_media:
        tmp_df = pred_df[pred_df["Media1"] == med]
        media1 = ["" if i >= 1 else x for i,x in enumerate(tmp_df["Media1"])]
        tmp_out = pd.DataFrame({"Main media": media1, "Second media": tmp_df["Media2"].tolist(), "Probability of growing": tmp_df["value"].tolist()})
        output_df.append(tmp_out)

    output_df = pd.concat(output_df)
    output_df.to_html('tmp.html')
    return pred_df

if __name__ == '__main__':
    main()