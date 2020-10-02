### Author Douwe Spaanderman - 23 June 2020 ###
import numpy as np
import pandas as pd
import argparse
import time
import json
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, accuracy_score
from sklearn import metrics

from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

# Currently use sys to get other script - in Future use package
import os
import sys
path_main = ("/".join(os.path.realpath(__file__).split("/")[:-2]))
sys.path.append(path_main + '/Utils/')
from help_functions import mean, str_to_bool, str_none_check

def load_data(Path, supplements=True, test_size=0.2, random_state=None):
    '''

    '''
    if not Path.endswith(".json"):
        Path = Path + "result.json"

    with open(Path) as json_file:
        data = json.load(json_file)

    if supplements == True:
        X = np.array(data['input_data_supplements'])
        Y = np.array(data['output_data_supplements'])
    elif supplements == False:
        X = np.array(data['input_data'])
        Y = np.array(data['output_data'])

    indices = np.arange(len(X))
    x_train, x_test, y_train , y_test, indices_train, indices_test = train_test_split(X, Y, indices,
                                                     test_size=test_size, random_state=random_state)

    return x_train, y_train, x_test, y_test, indices_train, indices_test

def grid_train_model(x_train, y_train, imbalance=False, random_state=None, cores=4, param_grid={'n_estimators': [100, 200, 500], 'max_features': ['auto', 'log2'], 'max_depth' : [None],'criterion' :['gini', 'entropy'], 'oob_score': [False], 'class_weight': ['balanced', 'balanced_subsample', None]}):
    '''

    '''
    if imbalance == False:
        model = RandomForestClassifier(random_state=random_state)
    elif imbalance == True:
        model = BalancedRandomForestClassifier(random_state=random_state)
    else:
        raise KeyError(f"Imbalance was provided with {imbalance}, please use False or True (bool)")

    # StratifiedKFold
    skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(estimator = model, param_grid = param_grid, 
                          cv = skf, n_jobs = cores, verbose = 2)

    grid_search.fit(x_train, y_train)
    print("Best paramaters using grid train")
    print(grid_search.best_params_)
    model = grid_search.best_estimator_

    return model

def train_model(x_train, y_train, imbalance=False, random_state=0, cores=4, n_estimators=100, max_features='Auto', max_depth=None, criterion='entropy', oob_score=False, class_weight='balanced'):
    '''

    '''
    if imbalance == False:
        model = RandomForestClassifier(random_state=random_state, n_estimators=n_estimators, max_features=max_features, max_depth=max_depth, criterion=criterion, oob_score=oob_score, class_weight=class_weight)
    elif imbalance == True:
        model = BalancedRandomForestClassifier(random_state=random_state, n_estimators=n_estimators, max_features=max_features, max_depth=max_depth, criterion=criterion, oob_score=oob_score, class_weight=class_weight)
    else:
        raise KeyError(f"Imbalance was provided with {imbalance}, please use False or True (bool)")
    
    model.fit(x_train, y_train)

    return model

def predict_model(x_test, y_test, model):
    '''

    '''
    predictions = model.predict(x_test)
    predictions_proba = model.predict_proba(x_test)

    print("Prediction for model")
    print(confusion_matrix(y_test, predictions))
    print('\n')
    print(classification_report(y_test, predictions))
    print('\n')
    print(f'Accuracy score: {accuracy_score(y_test, predictions)*100:.2f}%')
    print(f"Balanced accuracy score : {balanced_accuracy_score(y_test, predictions)*100:.2f}%")

    predictions = pd.DataFrame({"Predictions":predictions.tolist(), "True_labels":y_test.tolist(), "Probability_0": predictions_proba[:,0], "Probability_1":predictions_proba[:,1]})
    return predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Random Forest Method")
    parser.add_argument("Path", help="Path to json file")
    parser.add_argument("Save_location", help="Path where model and predictions are saved")
    parser.add_argument("-t", dest="train", nargs='?', default="single_train", help="Do you want to train or load a model", choices=['single_train', 'grid_train', 'load'])
    parser.add_argument("-g", dest="grid_param", nargs='?', default="Default", help="Default grid search or Path to json with grid_paramaters")
    parser.add_argument("-s", dest="supplements", nargs='?', default=True, help="Do you want to include supplements in prediction (default = True)", choices=['True', 'False'])
    parser.add_argument("-r", dest="random_state", nargs='?', default="0", help="Which random state do you want to select (default = 0)")
    parser.add_argument("-p", dest="partion", nargs='?', default=0.2, help="how do you want to partion the training/test data (default = 0.2)")
    parser.add_argument("-i", dest="imbalance", nargs='?', default=True, help="Do you want to use the imbalance library (default = True)", choices=['True', 'False'])
    parser.add_argument("-c", dest="cores", nargs='?', default=4, help="How many cores do you want to select (default = 4")
    parser.add_argument("-n", dest="n_estimators", nargs='?', default="100", help="Number of estimators/trees (default = 100)")
    parser.add_argument("-f", dest="max_features", nargs='?', default='auto', help="The number of features to consider when looking for the best split (default = auto), you can provide auto/sqrt/log2/none or any int/float")
    parser.add_argument("-d", dest="max_depth", nargs='?', default=None, help="max depth of tree (default = None)")
    parser.add_argument("-cr", dest="criterion", nargs='?', default="entropy", help="The function to measure the quality of a split (default = entropy)", choices=['gini', 'entropy'])
    parser.add_argument("-o", dest="oob_score", nargs='?', default="False", help="Number of estimators/trees (default = 100)")
    parser.add_argument("-w", dest="class_weight", nargs='?', default="balanced", help="Class_weight", choices=['balanced', 'balanced_subsample', 'None'])

    # Setup conditions
    args = parser.parse_args()
    start = time.time()
    if args.random_state == "None" or args.random_state == None:
        random_state = None
    elif type(args.random_state) == str:
        random_state = int(args.random_state)

    if args.max_depth == "None" or args.max_depth == None:
        max_depth = None
    elif type(args.max_depth) == str:
        max_depth = int(args.max_depth)

    if args.class_weight == "None":
        class_weight = None
    elif type(args.class_weight) == str:
        class_weight = str(args.class_weight)

    if args.max_features == "None":
        max_features = None
    elif args.max_features == "auto" or args.max_features == "sqrt" or args.max_features == "log2":
        max_features = args.max_features
    else: 
        max_features = int(args.max_features)

    if args.oob_score == "False":
        oob_score = False
    elif type(args.oob_score) == str:
        oob_score = int(args.oob_score)

    # Load data
    X_train, Y_train, X_test, Y_test, indices_train, indices_test = load_data(args.Path, supplements=str_to_bool(args.supplements), test_size=float(args.partion), random_state=random_state)

    # Train or load model (last currently not implemented)
    if args.train == "single_train":
        model = train_model(X_train, Y_train, imbalance=str_to_bool(args.imbalance), random_state=random_state, cores=int(args.cores), n_estimators=int(args.n_estimators), max_features=max_features, max_depth=max_depth, criterion=args.criterion, oob_score=oob_score, class_weight=class_weight)
    elif args.train == "grid_train":
        if args.grid_param == "Default":
            param_grid = {'n_estimators': [100, 200, 500], 'max_features': ['auto', 'log2'], 'max_depth' : [None],'criterion' :['gini', 'entropy'], 'oob_score': [False], 'class_weight': ['balanced', 'balanced_subsample', None]}
        else:
            try:
                if args.grid_param.endswith(".json"):
                    print("Currently not implemented")
                else:
                    print("Currently not implemented")
            except:
                raise KeyError("Unrecognized grid param ")
        
        print("These are the parameters to try using grid search")
        print(param_grid)
        model = grid_train_model(X_train, Y_train, imbalance=bool(args.imbalance), random_state=random_state, cores=int(args.cores), param_grid=param_grid)
    elif args.train == "load":
        print("Currently not implemented")
    else:
        raise KeyError(f"train was provided with {args.train}, please select either ['normal', 'grid', 'load']")

    # Prediction
    prediction = predict_model(X_test, Y_test, model=model)

    with open(args.Save_location + "indices_split_cache.json", 'w') as fp:
        json.dump({
            "indices_train":indices_train.tolist(),
            "indices_test":indices_test.tolist()
        }, fp)

    pickle.dump(model, open(args.Save_location + "RF_model.sav", 'wb'))
    prediction.to_pickle(args.Save_location + "Predictions.pkl")

    end = time.time()
