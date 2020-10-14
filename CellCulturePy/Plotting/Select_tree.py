import pandas as pd
import numpy as np
import pickle5 as pickle
import json
import os
from sklearn.tree import export_graphviz
from imblearn.ensemble import BalancedRandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import tree

# Currently use sys to get other script - in Future use package
import os
import sys
path_main = ("/".join(os.path.realpath(__file__).split("/")[:-2]))
sys.path.append(path_main + '/Utils/')
from help_functions import mean, str_to_bool, str_none_check

def treeplotting(model, cache=False, Path=False, save=False, show=True):
    '''

    '''
    if Path != False:
        if not Path.endswith("Model/"):
            Path = Path + "Model/"

        if not Path.endswith(".pkl"):
            Path = Path + "RF_model.sav"
        
        model = pd.read_pickle(Path)

    if cache != False:
        if not cache.endswith("Cache/"):
            cache = cache + "Cache/"

        if not Path.endswith(".json"):
            cache = cache + "cache_features.json"

        with open(cache) as json_file:
            data = json.loads(json_file.read())
            data = data["cache_all"] + data["cache_media"]

    dotfile = open("tree.dot", 'w')
    tree.export_graphviz(rf.estimators_[0], out_file = dotfile, feature_names = np.asarray(data))
    dotfile.close()

    if save != False:
        if not save.endswith("Figures/"):
            save = save + "Figures/"

        if not save.endswith(".png"):
            save = save + "TreeFig.png"

        os.system(f'dot -Tpng tree.dot -o t {save}')

    os.system('rm tree.dot')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot a descision tree from random forest")
    parser.add_argument("Path", help="Path to random forest sav output")
    parser.add_argument("Cache", help="Path to cache of features json")
    parser.add_argument("-s", dest="Save", nargs='?', default=False, help="location of file")
    parser.add_argument("-d", dest="Show", nargs='?', default=True, help="Do you want to show the plot?")

    args = parser.parse_args()
    start = time.time()
    print(bool(args.Show))
    Prediction_plotting(model=False, cache=args.Cache, Path=args.Path, save=args.Save, show=str_to_bool(args.Show))
    end = time.time()
    print('completed in {} seconds'.format(end-start))