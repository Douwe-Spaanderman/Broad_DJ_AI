### Author Douwe Spaanderman - 4 June 2020 ###

# This script downloads all the files from the filtered data frame

import pandas as pd
import os
import argparse
import time

def gsutil(link, Save="../../Data/Panel/."):
    '''
    Download gsutil link
    
    input:
    link = link to google bucket
    Save = location for item to be saved
    '''
    os.system(f'gsutil cp {link} {Save}')

def download_data(data, Path=False, Save="../../Data/Panel/.", Files="maf"):
    '''
    Main script to download data from google bucket

    input:
    data = is either a loaded pandas dataframe or a path to this csv file
    Path = use True if providing path for data
    Files = which files you would like to download from the gbucket
                pick either Panel or RNA
    '''
    if Path != False:
        if not Path.endswith(".pkl"):
            Path = Path + "filtered.pkl"
        data = pd.read_pickle(Path)

    if Files == "maf":
        pd.Series(data["PANEL_oncotated_maf_mutect2"].unique()).apply(gsutil, Save=Save)
    elif Files == "cnv":
        pd.Series(data["PANEL_cnv_calls"].unique()).apply(gsutil, Save=Save)
    elif Files == "RNA":
        print("currently not implemented")
    else:
        raise KeyError(f"Unrecognized file sort is used: {Files}. Please use either Panel or RNA")

    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download files from google bucket")
    parser.add_argument("Path", help="path to terra workspace filtered file")
    parser.add_argument("-s", dest="Save", nargs='?', default=False, help="location of file")
    parser.add_argument("-f", dest="Files", nargs='?', default='maf', help="which files would you like to download", choices=["maf", "cnv", "RNA", "WES"])

    args = parser.parse_args()
    start = time.time()
    data = download_data(data=False, Path=args.Path, Save=args.Save, Files=args.Files)
    end = time.time()
    print('completed in {} seconds'.format(end-start))