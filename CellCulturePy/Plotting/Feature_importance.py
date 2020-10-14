import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import time

# Currently use sys to get other script - in Future use package
import os
import sys
path_main = ("/".join(os.path.realpath(__file__).split("/")[:-2]))
sys.path.append(path_main + '/Utils/')
from help_functions import mean, str_to_bool, str_none_check

def Feature_importance(model, Path=False, save=False, show=True):
    '''

    '''
    if Path != False:
        if not Path.endswith("Model/"):
            Path = Path + "Model/"
            if not Path.endswith(".sav"):
                Path = Path + "RF_model.sav"
        
        model = pickle.load(open(Path, 'rb'))


    plt.rcParams["figure.figsize"] = 10,4

    x_p = np.linspace(-3,3, num=len(model.feature_importances_))
    y_p = model.feature_importances_

    fig, (ax,ax2) = plt.subplots(nrows=2, sharex=True)

    extent = [x_p[0]-(x_p[1]-x_p[0])/2., x_p[-1]+(x_p[1]-x_p[0])/2.,0,1]
    ax.imshow(y_p[np.newaxis,:], cmap="plasma", aspect="auto", extent=extent)
    ax.set_yticks([])
    ax.set_xlim(extent[0], extent[1])

    ax2.plot(x_p,y_p)

    plt.tight_layout()

    if save != False:
        if not save.endswith("Figures/"):
            save += "Figures/"
        plt.savefig(save + "Feature_importance.png")
    if show == True:
        plt.show()
    if show == False and save == False:
        warnings.warn('Both save and show is set to False')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Feature importance scoring")
    parser.add_argument("Path", help="Path to RF model")
    parser.add_argument("-s", dest="Save", nargs='?', default=False, help="location of file")
    parser.add_argument("-d", dest="Show", nargs='?', default=True, help="Do you want to show the plot?")

    args = parser.parse_args()
    start = time.time()
    print(bool(args.Show))
    Feature_importance(model=False, Path=args.Path, save=args.Save, show=str_to_bool(args.Show))
    end = time.time()
    print('completed in {} seconds'.format(end-start))