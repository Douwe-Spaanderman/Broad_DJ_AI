# Reading CNV files

import pandas as pd
import numpy as np
from pathlib import Path
import time
import argparse

# Currently use sys to get other script
import sys
sys.path.insert(1, "/Users/dspaande/Documents/GitProjects/Broad_DJ_AI/DeepGrowth/Utils/")
from help_functions import mean, str_to_bool, str_none_check

# Currently use sys to get other script
import sys
sys.path.insert(1, "/Users/dspaande/Documents/GitProjects/Broad_DJ_AI/DeepGrowth/Classes/")
from gene_one_hot import one_hot

def cnv_extract(cnv):
    '''

    '''
    file_name = cnv.stem.split('.')[0]
    cnv_frame = pd.read_csv(cnv, sep="\t")

    return cnv_frame, file_name

def filter_cnv(cnv, transform=None):
    '''

    '''
    chromosomelist = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13",   "14", "15", "16", "17", "18", "19", "20", "21", "22", "X", "Y"]                         
                                                                                            
    data = {}                                                                               
    for chromosoom in chromosomelist:
        cnv_tmp = cnv[cnv["Chromosome"] == chromosoom]
        if type(cnv_tmp) == pd.core.series.Series or cnv_tmp.empty == True:
            data[chromosoom] = 1
        elif len(cnv_tmp) > 1:
            data[chromosoom] = (sum(cnv_tmp["Num_Probes"] * cnv_tmp["Segment_Mean"]) / sum(cnv_tmp["Num_Probes"]))
        else:
            data[chromosoom] = cnv_tmp["Segment_Mean"].tolist()[0]

    return data

def onehot_cnv(cnv, file_name, idx):
    '''

    '''
    flat = one_hot(np.array(list(cnv.values())), np.array(list(cnv.keys())))
    flat.sanity()

    data = pd.DataFrame(data={'File': file_name, 'cnv_one_hot': flat}, index=[idx])

    return data

def main_cnv(directory="../../Data/Panel/", transform=None, Save=False, Show=True):
    '''

    '''
    if directory.endswith("/"):
        pathlist = Path(directory).glob('*.tumor.called*')
        number_of_files = len(list(pathlist))
        pathlist = Path(directory).glob('*.tumor.called*')
    else:
        pathlist = Path(directory).glob('/*.tumor.called*')
        number_of_files = len(list(pathlist))
        pathlist = Path(directory).glob('/*.tumor.called*')

    data = []
    for idx, path in enumerate(pathlist):
        print(path)
        cnv_frame, file_name = cnv_extract(path)
    
        print("now doing: {}".format(file_name))
        cnv_frame = filter_cnv(cnv_frame, transform=transform)
        cnv_frame = onehot_cnv(cnv_frame, file_name, idx)

        data.append(cnv_frame)
        if idx % 10 == 0:
            print(f'done {idx+1} out of {number_of_files}')

    data = pd.concat(data)

    if Save != False:
        if not Save.endswith(".pkl"):
            Save = Save + "cnv_extract.pkl"
            
        data.to_pickle(Save)

    if Show == True:
        print(data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Read and combine all cnv data with the data")
    parser.add_argument("Path", help="path to directory with cnv files")
    parser.add_argument("-t", dest="Transform", nargs='?', default=None, help="would you like to transform data (normal data is log2 transformed)", choices=['None', 'backtransformed'])
    parser.add_argument("-s", dest="Save", nargs='?', default=False, help="location of file")
    parser.add_argument("-d", dest="Show", nargs='?', default=True, help="Do you want to show the plot?")

    args = parser.parse_args()
    start = time.time()
    main_cnv(directory=args.Path, Save=args.Save, Show=str_to_bool(args.Show), transform=str_none_check(args.Transform))
    end = time.time()
    print('completed in {} seconds'.format(end-start))