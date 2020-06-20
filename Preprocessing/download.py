### Author Douwe Spaanderman - 4 June 2020 ###

# This script downloads all the files from the filtered data frame

import pandas as pd
import os

def gsutil(link, Save="../Data/Panel/."):
    '''

    '''
    os.system(f'gsutil cp {link} {Save}')

def download_data(data, Path=False, Save="../Data/Panel/.", Files="Panel"):
    '''
    Main script to run the changing of media. First creates

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

    if Files == "Panel":
        pd.Series(data["PANEL_oncotated_maf_mutect2"].unique()).apply(gsutil, Save=Save)
    elif Files == "RNA":
        print("currently not implemented")
    else:
        raise KeyError(f"Unrecognized file sort is used: {Files}. Please use either Panel or RNA")

    return data

download_data(data=False, Path="../Data/Ongoing/")