import pandas as pd
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
import warnings

from sklearn.metrics import classification_report, roc_curve, roc_auc_score, precision_recall_curve, auc
# 

# Confusion Matrix

# ROC curve
## some notes : Not super usefull as ROC curves are more important with roughly equal number of observations for each class
def ROC(data, save=False, show=True):
    '''

    '''
    fpr, tpr, _ = roc_curve(data.iloc[:,1],  data.iloc[:,3])
    auc = round(roc_auc_score(data.iloc[:,1], data.iloc[:,3]), 2)
    plt.plot([0,1],[0,1], linestyle='--')
    plt.plot(fpr,tpr,label="Imbalanced Random Forest="+str(auc), marker='.')
    plt.legend(loc=4)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    
    if save != False:
        plt.savefig(save + "ROC.png")
    if show == True:
        plt.show()
    if show == False and save == False:
        warnings.warn('you are not checking the input data for media/supplements')



# Precision-Recall Curves
## More usefull due to less imbalance
def Precision_Recall(data, save=False, show=True):
    '''

    '''
    precision, recall, _ = precision_recall_curve(data.iloc[:,1], data.iloc[:,3])
    auc_value = round(auc(recall, precision), 2)

    # plot the precision-recall curves
    line = data.iloc[:,1]
    line = len(line[line==1]) / len(line)
    plt.plot([0, 1], [line, line], linestyle='--')
    plt.plot(recall, precision,label="Imbalanced Random Forest="+str(auc_value), marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()

    if save != False:
        plt.savefig(save + "Precision_Recall.png")
    if show == True:
        plt.show()
    if show == False and save == False:
        warnings.warn('Both save and show is set to False')

def Prediction_plotting(data, Path=False, save=False, show=True):
    '''

    '''
    if Path != False:
        if not Path.endswith(".pkl"):
            Path = Path + "Predictions.pkl"
        
        data = pd.read_pickle(Path)

    
    if save != False:
        if save.endswith(".png"):
            raise KeyError("Please remove specified name file and use path instead for saving figures")
    
    Precision_Recall(data, save=save, show=show)
    ROC(data, save=save, show=show)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prediction vs true label plotting")
    parser.add_argument("Path", help="Path to pandas output")
    parser.add_argument("-s", dest="Save", nargs='?', default=False, help="location of file")
    parser.add_argument("-d", dest="Show", nargs='?', default=True, help="Do you want to show the plot?")

    args = parser.parse_args()
    start = time.time()
    print(bool(args.Show))
    Prediction_plotting(data=False, Path=args.Path, save=args.Save, show=bool(args.Show))
    end = time.time()
    print('completed in {} seconds'.format(end-start))