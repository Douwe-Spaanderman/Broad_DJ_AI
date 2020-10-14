import argparse
import time
import warnings
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import plot_model
import tensorflow as tf

# Currently use sys to get other script - in Future use package
import os
import sys
path_main = ("/".join(os.path.realpath(__file__).split("/")[:-2]))
sys.path.append(path_main + '/Utils/')
from help_functions import mean, str_to_bool, str_none_check

def plot_neural(data, Path=False, save=False, show=True):
    '''

    '''
    if Path != False:
        model = tf.keras.models.load_model(Path)

    if save != False:
        if save.endswith(".png"):
            raise KeyError("Please remove specified name file and use path instead for saving figures")
        elif not save.endswith("Figures/"):
            save += "Figures/"

        save += "NeuralDisplay.png"
            
        plot_model(model, to_file=save, show_shapes=True, show_layer_names=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prediction vs true label plotting")
    parser.add_argument("Path", help="Path to pandas output")
    parser.add_argument("-s", dest="Save", nargs='?', default=False, help="location of file")
    parser.add_argument("-d", dest="Show", nargs='?', default=True, help="Do you want to show the plot?")

    args = parser.parse_args()
    start = time.time()
    print(bool(args.Show))
    Prediction_plotting(data=False, Path=args.Path, save=args.Save, show=str_to_bool(args.Show))
    end = time.time()
    print('completed in {} seconds'.format(end-start))