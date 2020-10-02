"""
Demonstrates plotting chromosome ideograms and genes (or any features, really)
using matplotlib.
1) Assumes a file from UCSC's Table Browser from the "cytoBandIdeo" table,
saved as "ideogram.txt". Lines look like this::
    #chrom  chromStart  chromEnd  name    gieStain
    chr1    0           2300000   p36.33  gneg
    chr1    2300000     5300000   p36.32  gpos25
    chr1    5300000     7100000   p36.31  gneg
2) Assumes another file, "ucsc_genes.txt", which is a BED format file
   downloaded from UCSC's Table Browser. This script will work with any
   BED-format file.
"""

from matplotlib import pyplot as plt
from matplotlib.collections import BrokenBarHCollection
import pandas
import argparse
import warnings
import time

import os
import sys
path_main = ("/".join(os.path.realpath(__file__).split("/")[:-2]))
sys.path.append(path_main + '/Utils/')
from help_functions import mean, str_to_bool, str_none_check

# Here's the function that we'll call for each dataframe (once for chromosome
# ideograms, once for genes).  The rest of this script will be prepping data
# for input to this function
#
def chromosome_collections(df, y_positions, height,  **kwargs):
    """
    Yields BrokenBarHCollection of features that can be added to an Axes
    object.
    Parameters
    ----------
    df : pandas.DataFrame
        Must at least have columns ['chrom', 'start', 'end', 'color']. If no
        column 'width', it will be calculated from start/end.
    y_positions : dict
    Keys are chromosomes, values are y-value at which to anchor the
        BrokenBarHCollection
    height : float
        Height of each BrokenBarHCollection
        Additional kwargs are passed to BrokenBarHCollection
    """
    del_width = False
    if 'width' not in df.columns:
        del_width = True
        df['width'] = df['end'] - df['start']
    for chrom, group in df.groupby('chrom'):
        print(chrom)
        yrange = (y_positions[chrom], height)
        xranges = group[['start', 'width']].values
        yield BrokenBarHCollection(
            xranges, yrange, facecolors=group['colors'], **kwargs)
    if del_width:
        del df['width']

def main_pred(Path=False, save=False, show=True):
    '''

    '''
    if Path != False:
        if not Path.endswith(".txt"):
            Path = Path + "ideogram.txt"
    else:
        Path = path_main + "/Data/ideogram.txt"

    # Height of each ideogram
    chrom_height = 1.2

    # Spacing between consecutive ideograms
    chrom_spacing = 1

    # Height of the gene track. Should be smaller than `chrom_spacing` in order to
    # fit correctly
    gene_height = 0.2

    # Padding between the top of a gene track and its corresponding ideogram
    gene_padding = 0.1

    # Width, height (in inches)
    figsize = (20,10)

    # Decide which chromosomes to use
    chromosome_list = ['chr%s' % i for i in list(range(1, 23)) + ['X', 'Y']]

    # Keep track of the y positions for ideograms and genes for each chromosome,
    # and the center of each ideogram (which is where we'll put the ytick labels)
    ybase = 0
    chrom_ybase = {}
    gene_ybase = {}
    chrom_centers = {}

    # Iterate in reverse so that items in the beginning of `chromosome_list` will
    # appear at the top of the plot
    for chrom in chromosome_list[::-1]:
        chrom_ybase[chrom] = ybase
        chrom_centers[chrom] = ybase + chrom_height / 2.
        gene_ybase[chrom] = ybase - gene_height - gene_padding
        ybase += chrom_height + chrom_spacing

    # Read in ideogram.txt, downloaded from UCSC Table Browser
    ideo = pandas.read_table(
        Path,
        skiprows=1,
        names=['chrom', 'start', 'end', 'name', 'gieStain']
    )

    # Filter out chromosomes not in our list
    ideo = ideo[ideo.chrom.apply(lambda x: x in chromosome_list)]

    # Add a new column for width
    ideo['width'] = ideo.end - ideo.start

    # Colors for different chromosome stains
    color_lookup = {
        'gneg': (1., 1., 1.),
        'gneg': (.7, .7, .7),
        'gpos50': (.4, .4, .4),
        '75': (.2, .2, .2),
        'gpos100': (0., 0., 0.),
        'gpos5': (.8, .4, .4),
        'g': (.8, .8, .8),
        'g': (.9, .9, .9),
    }

    # Add a new column for colors
    ideo['colors'] = ideo['gieStain'].apply(lambda x: color_lookup[x])

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # Now all we have to do is call our function for the ideogram data...
    for collection in chromosome_collections(ideo, chrom_ybase, chrom_height):
        ax.add_collection(collection)

    # Axes tweaking
    ax.set_yticks([chrom_centers[i] for i in chromosome_list])
    ax.set_yticklabels(chromosome_list, fontsize=14)
    ax.axis('tight')
    ax.set_xticks([])
    
    if save != False:
        plt.savefig(save + "targeted_overview.png")
    if show == True:
        plt.show()
    if show == False and save == False:
        warnings.warn('you are not checking the input data for media/supplements')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plotting targeted sequence panel")
    parser.add_argument("-p", dest="Path", help="path to ideogram.txt")
    parser.add_argument("-s", dest="Save", nargs='?', default=False, help="location of save")
    parser.add_argument("-d", dest="Show", nargs='?', default=True, help="Do you want to show the plot?")

    args = parser.parse_args()
    start = time.time()
    main_pred(Path=args.Path, save=args.Save, show=str_to_bool(args.Show))
    end = time.time()
    print('completed in {} seconds'.format(end-start))