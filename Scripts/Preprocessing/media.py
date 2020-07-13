### Author Remi Marenco and Douwe Spaanderman - 1 June 2020 ###

# This script creates a changed view on media into classes of medium and supplements

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import re
import argparse
import time
import warnings

# Currently use sys to get other script
import sys
sys.path.insert(1, "../Classes/")
from media_class import Medium, Supplement, GrowthMedium, Medium_one_hot, Supplement_one_hot, GrowthMedium_one_hot

# Split by ',', it will give all media. Then if one has '_' it contains supplement
# We can have more than 50/50...and not always 50/50. Maybe we could remove them if not enough data
cache_media_result: dict = {}
def extract_media(media_content: str):
    if pd.isna(media_content):
        return GrowthMedium()
        
    if media_content in cache_media_result:
        return cache_media_result[media_content]

    current_growth_medium = GrowthMedium()
        
    if len(re.split('_|,',media_content)) == 1:
        # Also check if percentage afterwards
        media_content = media_content.split(":")
        if len(media_content) == 1:
            media_content = media_content[0]
            new_media = Medium(media_content, "100%")
            current_growth_medium.media[media_content] = new_media
        else:
            new_media = Medium(media_content[0], media_content[1]) 
            current_growth_medium.media[media_content[0]] = new_media
            
        current_growth_medium.clean_media()
        return current_growth_medium
        
    # Handle the case => RETM:50%,RETM_SMGM: RETM 50%,RETM_SMGM: SMGM 50%,RETM_SMGM:100%,SMGM:50%
    if media_content == "RETM:50%,RETM_SMGM: RETM 50%,RETM_SMGM: SMGM 50%,RETM_SMGM:100%,SMGM:50%":
        current_growth_medium.media["RETM"] = Medium("RETM", "50%")
        current_growth_medium.media["SMGM"] = Medium("SMGM", "50%")
        
    # Handle the case => AR5,SMGM
    if media_content in ["AR5,BEGM", "AR5,SMGM", "CM,Pancreas Organoid", "AR5,CM", "CM,SMGM", "BEGM,SMGM"]:
        media_content = media_content.split(",")
        name = media_content[0]
        current_growth_medium.media[name] = Medium(name, "50%")
        name = media_content[1]
        current_growth_medium.media[name] = Medium(name, "50%")
        return current_growth_medium
        
    # Split to extract each string medium+percentage(+supplement)
    list_media_supplements = media_content.split(',')
        
    # Supplements are separated with _, need to separate them in the list
    regexp_supplement_extraction = "([^_]*)_(.*)"
    pattern_supplement_extraction = re.compile(regexp_supplement_extraction)
        
    list_media_and_some_supplements = []
    list_supplements = []
    for media_supplement in list_media_supplements:
        # Special case for errors with RGid estradio close to OPAC
        # Exemple: AR5:50%,B27:1U/L_¿-estradiol: 50.0 %
        if 'estradiol' in media_supplement:
            if '_' in media_supplement:
                # We split and transform estradiol to OPAC then the other one is a supplement
                supplement_false_media_list = media_supplement.split('_')
                    
                media_to_replace = supplement_false_media_list[1]
                    
                if '%' in media_to_replace:
                    media_to_replace = media_to_replace.replace('¿-estradiol', 'OPAC')
                    
                    supplement = supplement_false_media_list[0]
                    media_supplement = '_'.join([media_to_replace, supplement])
            elif '%' in media_supplement:
                import pdb; pdb.set_trace()
                # The media is just OPAC with the %age
                media_supplement = media_supplement.replace('¿-estradiol', 'OPAC')
            
        if "%_" in media_supplement:
            media_and_supplement_list = media_supplement.split("%_")
                
                
            list_media_and_some_supplements.append(media_and_supplement_list[0] + "%")
            list_supplements.append(media_and_supplement_list[1])
        else:
            list_media_and_some_supplements.append(media_supplement)

    # Regexp to get the values out
    regexp_medium_percentage_supplement = "(.*)[:|\ ](\d+\.?\d+[^:]*)$"
    pattern = re.compile(regexp_medium_percentage_supplement)

    for media_supplement in list_media_and_some_supplements:
        result_search = pattern.search(media_supplement)
            
        # It means our pattern was not appropriate, like B27:1U/L
        if not result_search:
            result_search = re.compile("(.*):(.*)").search(media_supplement)

        name, amount = result_search.group(1), result_search.group(2)
        # Cleaning name of the ":" and " "
        name = name.replace(":", " ").strip()  

        if '%' in amount:
            # Media
            new_media = Medium(name, amount)
            current_growth_medium.media[name] = new_media
        else:
            # Supplement
            # TODO: Sometimes amount can have more than just the amount
            new_element = Supplement(name, amount)
            current_growth_medium.supplements[name] = new_element
        
    if list_supplements is not None:
        if list_supplements:
            for supplement in list_supplements:

                # TODO: Use function
                result_search = pattern.search(supplement)

                # It means our pattern was not appropriate, like B27:1U/L
                if not result_search:
                    result_search = re.compile("(.*):(.*)").search(media_supplement)
                    
                name, amount = result_search.group(1), result_search.group(2)
                # Cleaning name of the ":" and " "
                name = name.replace(":", " ").strip()

                new_element = Supplement(name, amount)
                current_growth_medium.supplements[name] = new_element

            # We could clean here. If 100% but more than 1 media issue. If 50% but more than 2 media issue, etc...
            current_growth_medium.clean_media()

            cache_media_result[media_content] = current_growth_medium

    return current_growth_medium

def one_hot_media(media_content: GrowthMedium, all_medium:list, all_supplements:list, percentage=False):
    '''
    Creates one_hot encoding for media either by 'real' one hot for example: [0,1,0,1,0,...] or based on
    the percentage [0,1,0,0,0.5,...] if percentage is pased as True.

    Input
    media_content = GrowthMedium class
    all_medium = sorted list of all medium occuring in the dataset
    all_supplements = sorted list of all supplements occuring in the dataset
    percentage = to include percentages in one-hot encoding or not

    Output:
    GrowthMedium_one_hot = one-hot encoding of media and supplements
    '''
    #Create empty numpy arrays
    media_array = np.zeros(shape=(len(all_medium)))
    suple_array = np.zeros(shape=(len(all_supplements)))

    if percentage == False:
        # Get all keys
        media_types = set(media_content.media.keys())
        suple_types = set(media_content.supplements.keys())

        #Get index
        index_media = [i for i, item in enumerate(all_medium) if item in media_types]
        index_suple = [i for i, item in enumerate(all_supplements) if item in suple_types]
        
        #Change 0 -> 1 for index
        np.put(media_array, index_media, 1)
        np.put(suple_array, index_suple, 1)

        #Create class
        current_one_hot = GrowthMedium_one_hot(
            Medium_one_hot(media_array, all_medium),
            Supplement_one_hot(suple_array, all_supplements)
        )

    elif percentage == True:
        # Get all dictionaries
        media_types = media_content.media
        suple_types = media_content.supplements

        # loop through media
        for key, value in media_types.items():
            index_media = all_medium.index(key)
            #Change way percentages are represented
            percentage = value.amount
            if percentage.endswith('%'):
                percentage = float(percentage[:-1])/100
            else:
                percentage = float(1)
                warnings.warn('No percentage was provided.. returning 1')

            np.put(media_array, index_media, percentage)

        # loop through supplements -> currenlty unable to do the same as with media due to different volumes (not percentages)
        index_suple = [i for i, item in enumerate(all_supplements) if item in suple_types]
        np.put(suple_array, index_suple, 1)

        #Create class
        current_one_hot = GrowthMedium_one_hot(
            Medium_one_hot(media_array, all_medium),
            Supplement_one_hot(suple_array, all_supplements)
        )
    else:
        raise KeyError(f"Wierd instance of percentage option was given for one_hot_media: {percentage}")

    return current_one_hot

def main_media(data, Path=False, Save=False, Encoding="Extract"):
    '''
    Main script to run the changing of media. First creates

    input:
    data = is either a loaded pandas dataframe or a path to this csv file
    Path = use True if providing path for data
    Save = if and where to save
    Encoding = which media representation you would like to have
                pick either Extract, One-hot or One-hot-percentage
    '''
    if Path != False:
        if not Path.endswith(".pkl"):
            Path = Path + "filtered.pkl"
        data = pd.read_pickle(Path)

    data["media_class"] = data["Media_type"].apply(extract_media)

    if Encoding == "One-hot":
        all_media = sorted(list(set().union(*(d.keys() for d in [x.media for x in data["media_class"]]))))
        all_supplements = sorted(list(set().union(*(d.keys() for d in [x.supplements for x in data["media_class"]]))))

        data["one-hot"] = data["media_class"].apply(one_hot_media, all_medium=all_media, all_supplements=all_supplements)

    elif Encoding == "One-hot-percentage":
        all_media = sorted(list(set().union(*(d.keys() for d in [x.media for x in data["media_class"]]))))
        all_supplements = sorted(list(set().union(*(d.keys() for d in [x.supplements for x in data["media_class"]]))))

        data["one-hot"] = data["media_class"].apply(one_hot_media, all_medium=all_media, all_supplements=all_supplements, percentage=True)

    if Save != False:
        if not Save.endswith(".pkl"):
            Save = Save + "after_media.pkl"
        #data.to_csv(save_l, encoding = "utf-8", sep="\t")
        data.to_pickle(Save)

    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Change media encoding")
    parser.add_argument("Path", help="path to terra workspace filtered file")
    parser.add_argument("-s", dest="Save", nargs='?', default=False, help="location of file")
    parser.add_argument("-e", dest="Encoding", nargs='?', default='Extract', help="which representation of media you would like", choices=["Extract", "One-hot", "One-hot-percentage"])

    args = parser.parse_args()
    start = time.time()
    data = main_media(data=False, Path=args.Path, Save=args.Save, Encoding=args.Encoding)
    end = time.time()
    print('completed in {} seconds'.format(end-start))