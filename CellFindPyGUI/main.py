### Author Douwe Spaanderman - 29 September 2020 ###
import os
from gooey import GooeyParser

# Currently use sys to get other script
import sys
sys.path.insert(1, "/Users/dspaande/Documents/GitProjects/Broad_DJ_AI/CellFindPyGUI/Classes/")
from media_class import Medium, Supplement, GrowthMedium, Medium_one_hot, Supplement_one_hot, GrowthMedium_one_hot
from gene_one_hot import one_hot

# Currently use sys to get other script
sys.path.insert(1, "/Users/dspaande/Documents/GitProjects/Broad_DJ_AI/CellFindPyGUI/Utils/")
from help_functions import mean, str_to_bool, str_none_check

def main(tumor, tissue):
    '''

    '''
    print(tumor)
    print(tissue)

if __name__ == '__main__':
    parser = GooeyParser(description="GUI for predicting media conditions")
    parser.add_argument("Tumor", help="what is your tumor type")
    parser.add_argument("Tissue", help="what is your tissue type")
    #parser.add_argument("-n", dest="DiseaseName", nargs='?', default=None, help="Do you want to include disease name?", choices=['Highest level', 'Lowest level', 'None'])
    #parser.add_argument("-t", dest="Tissue", nargs='?', default=False, help="Do you want to include tissue site?")
    #parser.add_argument("-r", dest="Dimension", nargs='?', default=False, help="Do you want to include dimension?")
    #parser.add_argument("-s", dest="Save", nargs='?', default=False, help="location of file")
    #parser.add_argument("-d", dest="Show", nargs='?', default=True, help="Do you want to show the plot?")

    args = parser.parse_args()
    main(tumor=args.Tumor, tissue=args.Tissue)