### Author Douwe Spaanderman - 29 September 2020 ###
import os
from gooey import Gooey, GooeyParser
from message import display_message

# Currently use sys to get other script
#import sys
#sys.path.insert(1, "/Users/dspaande/Documents/GitProjects/Broad_DJ_AI/CellFindPyGUI/Classes/")
#from media_class import Medium, Supplement, GrowthMedium, Medium_one_hot, Supplement_one_hot, GrowthMedium_one_hot
#from gene_one_hot import one_hot

# Currently use sys to get other script
#sys.path.insert(1, "/Users/dspaande/Documents/GitProjects/Broad_DJ_AI/CellFindPyGUI/Utils/")
#from help_functions import mean, str_to_bool, str_none_check

@Gooey(dump_build_config=True, program_name="CellCulturePy")
def main():
    '''

    '''
    parser = GooeyParser(description="Predicting media conditions with genomic profile")
    parser.add_argument("TumorType", help="what is your tumor type", choices=['yes', 'no'])
    parser.add_argument("Tissue", help="what is your tissue type", choices=['yes', 'no'])
    parser.add_argument("Dimension", help="what is your growing dimension", choices=['2D', '3D', 'Suspension'])
    parser.add_argument("maf", help="Select maf file from TWIST (mutect1 or 2)", widget="FileChooser")
    parser.add_argument("cnv", help="Select cnv file from TWIST (.tumor.called)", widget="FileChooser")
    parser.add_argument("-m", dest="Media", nargs='+', default=None, choices=['yes', 'no'], help="you can select one or multiple media types you want to look for", widget="Listbox")
    #parser.add_argument("-t", dest="Tissue", nargs='?', default=False, help="Do you want to include tissue site?")
    #parser.add_argument("-r", dest="Dimension", nargs='?', default=False, help="Do you want to include dimension?")
    #parser.add_argument("-s", dest="Save", nargs='?', default=False, help="location of file")
    #parser.add_argument("-d", dest="Show", nargs='?', default=True, help="Do you want to show the plot?")

    parser.parse_args()
    display_message()

if __name__ == '__main__':
    main()