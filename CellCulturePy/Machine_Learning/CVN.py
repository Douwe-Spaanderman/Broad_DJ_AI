### Author Douwe Spaanderman - 22 June 2020 ###
import numpy as np
import argparse
import time
import json
import pandas as pd

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, accuracy_score
from hyperopt import Trials, STATUS_OK, tpe
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.models import Sequential
from hyperas import optim
from hyperas.distributions import choice, uniform

def data(Path):
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """
    if not Path.endswith(".json"):
        Path = Path + "result.json"

    with open(Path) as json_file:
        data = json.load(json_file)

    X = np.array(data['input_data_supplements'])
    y = np.array(data['output_data_supplements'])

    x_train, x_test, y_train , y_test = train_test_split(X, y, 
                                                     test_size=0.2, random_state=0)

    return x_train, y_train, x_test, y_test

def create_model(x_train, y_train, x_test, y_test):
    """
    Model providing function:
    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=5, input_shape=(6, 45)))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    # If we choose 'two', add an additional second layer
    if {{choice(['one', 'two'])}} == 'two':
        model.add(Dense({{choice([16, 32, 64, 128, 256])}}))
        model.add(Activation({{choice(['relu', 'sigmoid'])}}))
        model.add(Dropout({{uniform(0, 1)}}))
    

        # If we choose 'three', add an additional thirth layer
        if {{choice(['two', 'three'])}} == 'three':
            model.add(Dense({{choice([8, 16, 32, 64, 128, 256])}}))

            # We can also choose between complete sets of layers

            model.add({{choice([Dropout(0.5), Activation('linear')])}})
            model.add(Activation('relu'))

    model.add(Flatten())
    # If we choose 'four', add an additional fourth layer
    if {{choice(['three', 'four'])}} == 'four':
        model.add(Dense({{choice([8, 16, 32, 64])}}))

        # We can also choose between complete sets of layers

        model.add({{choice([Dropout(0.5), Activation('linear')])}})
        model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                  optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})

    result = model.fit(x_train, y_train,
              batch_size={{choice([64, 128])}},
              epochs=1,
              verbose=2,
              validation_split=0.1)
    
    validation_acc = np.amax(result.history['val_accuracy']) 
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}

def predict_model(x_test, y_test, model):
    '''

    '''
    predictions = model.predict(x_test)
    
    predictions = [0 if x < 0.5 else 1 for x in predictions]
    print(confusion_matrix(y_test, predictions))
    print('\n')
    print(classification_report(y_test, predictions))
    print('\n')
    print(f'Accuracy score: {accuracy_score(y_test, predictions)*100:.2f}%')
    print(f"Balanced accuracy score : {balanced_accuracy_score(y_test, predictions)*100:.2f}%")

    predictions = pd.DataFrame({"Predictions":predictions, "True_labels":y_test.tolist()})
    return predictions

def main_neural(path, save=False):
    '''

    '''
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=1,
                                          trials=Trials(),
                                          data_args=(path,)
    )

    print(args.Path)

    X_train, Y_train, X_test, Y_test = data(path)
    #X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)

    best_model.summary()
    
    print('\n')
    predictions = predict_model(X_test, Y_test, model=best_model)

    if save != False:
        best_model.save(args.Save_location + "MLP_model")
        predictions.to_pickle(args.Save_location + "Predictions.pkl")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multilayer perceptron")
    parser.add_argument("Path", help="path to terra workspace file")
    parser.add_argument("Save_location", help="Path where model and predictions are saved")
    
    args = parser.parse_args()
    start = time.time()

    main_neural(path=args.Path, save=args.Save_location)

    end = time.time()
    print('completed in {} seconds'.format(end-start))