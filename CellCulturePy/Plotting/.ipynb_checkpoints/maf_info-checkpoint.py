### Author Douwe Spaanderman - 16 June 2020 ###
import pandas as pd
import argparse
import time

def plotting(data, Scale="log2", save=False, show=True, extention=""):
    '''
    
    '''
    



    return data

def maf_info(data, Path=False, Scale="log2", Save=False, Show=True):
    '''

    '''
    if Path != False:
        if not Path.endswith(".pkl"):
            Path = Path + "maf_extract.pkl"
        
        data = pd.read_pickle(Path)

    plotting(data)

    from collections import Counter 
    print(Counter(list(data["Hugo_Symbol"])))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Does nothing yet but Counter")
    parser.add_argument("Path", help="path to terra workspace filtered file")
    parser.add_argument("-s", dest="Save", nargs='?', default=False, help="location of file")
    parser.add_argument("-d", dest="Show", nargs='?', default=True, help="Do you want to show the plot?")
    parser.add_argument("-c", dest="Data_scaling", nargs='?', default="log2", help="how would you like to scale the data", choices=["log2", "normalized", "log10"])

    args = parser.parse_args()
    start = time.time()
    maf_info(data=False, Path=args.Path, Scale=args.Data_scaling, Save=args.Save, Show=args.Show)
    end = time.time()
    print('completed in {} seconds'.format(end-start))