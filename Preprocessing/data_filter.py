### Author Douwe Spaanderman - 27 May 2020 ###

# This script runs the whole prefiltering of Terra workspace data depending on which Arguments are inserted

# Libraries
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import time

def read_terra(input_loc):
    '''
    read the tsv file from terra, make string columns for nan

    input:
    input_loc = Path object with the location

    output:
    terra workspace data
    '''
    data = pd.read_csv(input_loc, sep="\t", index_col=0)
    data["PANEL_renamed_bai_file"] = [str(x) for x in data["PANEL_renamed_bai_file"]]
    data["RNA_fastq1"] = [str(x) for x in data["RNA_fastq1"]]
    
    return data

def filter_nan_sequence(data, selection="PANEL"):
    '''
    Filters out all rows which have NaN values for the selected sequencing information

    input:
    data = terra workspace data
    selection = any of the following: Panel/Wes/RNA/All

    output:
    filtered dataframe with only compleet cases
    '''
    if type(selection) == str:
        if selection.upper() == "PANEL":
            data = data[data["PANEL_renamed_bai_file"] != 'nan']
        elif selection.upper() == "RNA":
            data = data[data["RNA_fastq1"] != 'nan']
        elif selection.upper() == "WES":
            raise SyntaxError("Currently not implemented")
        elif selection.upper() == "ALL":
            data = data[(data["PANEL_renamed_bai_file"] != 'nan') & (data["RNA_fastq1"] != 'nan')]
        else:
            raise NameError("Implemented unused argument for filtering data, input was {}, but should be panel/wes/rna/all".format(selection))
    else:
        raise TypeError("filter_nan_sequencing was provided with a {}, while it should be a string".format(type(selection)))

    return data

def main_filter(Path, Save=False, Filter="PANEL"):
    '''
    Main script to run data filtering

    input:
    Path = location to terra workspace file
    Save = If to save and where (default is False)
    Filter = any of the following: Panel/Wes/RNA/All (default is Panel)

    output:
    filtered dataframe with only compleet cases
    '''

    data = read_terra(input_loc = Path)
    data = filter_nan_sequence(data=data, selection=Filter)

    if Save != False:
        if not Save.endswith(".pkl"):
            Save = Save + "filtered.pkl"
        
        data.to_pickle(Save)

    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="filter data")
    parser.add_argument("Path", help="path to terra workspace file")
    parser.add_argument("-s", dest="Save", nargs='?', default=False, help="location of file")
    parser.add_argument("-f", dest="Filter", nargs='?', default='PANEL', help="which sequencing you want to select (RNA/WES/PANEL)")

    args = parser.parse_args()
    start = time.time()
    data = main_filter(Path=args.Path, Save=args.Save, Filter=args.Filter)
    end = time.time()
    print('completed in {} seconds'.format(end-start))