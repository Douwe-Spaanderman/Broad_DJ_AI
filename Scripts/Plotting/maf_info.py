### Author Douwe Spaanderman - 16 June 2020 ###

import pandas as pd

def plotting(data, Scale="log2", save=False, show=True, extention=""):
    '''
    
    '''
    



    return data

def maf_info(data, Path, Scale="log2", save=False, show=True):
    '''

    '''
    if Path != False:
        if not Path.endswith(".pkl"):
            Path = Path + "maf_extract.pkl"
        
        data = pd.read_pickle(Path)

    plotting(data)

    from collections import Counter
    print(Counter(list(data["Hugo_Symbol"])))

maf_info(data=False, Path="../Data/Ongoing/")