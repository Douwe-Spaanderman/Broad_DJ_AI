### Author Douwe Spaanderman - 22 June 2020 ###
import numpy as np
import argparse
import time
import json

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, accuracy_score
from hyperopt import Trials, STATUS_OK, tpe
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.models import Sequential
from hyperas import optim
from hyperas.distributions import choice, uniform

def data():
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """
    with open('../Data/Ongoing/result.json') as json_file:
        data = json.load(json_file)

    X = np.array(data['input_data_supplements'])
    y = np.array(data['output_data_supplements'])

    x_train, x_test, y_train , y_test = train_test_split(X, y, 
                                                     test_size=0.2, random_state=0)

    return x_train, y_train, x_test, y_test

#def data(Path='../Data/Ongoing/result.json'):
#    """
#    Data providing function:
#
#    This function is separated from create_model() so that hyperopt
#    won't reload data for each evaluation run.
#    """
#    if not Path.endswith(".json"):
#        Path = Path + "result.json"
#
#    with open(Path) as json_file:
#        data = json.load(json_file)
#
#    X = np.array(data['input_data_supplements'])
#    y = np.array(data['output_data_supplements'])
#
#    x_train, x_test, y_train , y_test = train_test_split(X, y, 
#                                                     test_size=0.2, random_state=0)

#    return x_train, y_train, x_test, y_test

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
    model.add(Dense(512, input_shape=(len(x_train[0]),)))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([256, 512, 1024])}}))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(Dropout({{uniform(0, 1)}}))

    # If we choose 'four', add an additional fourth layer
    if {{choice(['three', 'four'])}} == 'four':
        model.add(Dense(100))

        # We can also choose between complete sets of layers

        model.add({{choice([Dropout(0.5), Activation('linear')])}})
        model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                  optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})

    result = model.fit(x_train, y_train,
              batch_size={{choice([64, 128])}},
              epochs=100,
              verbose=2,
              validation_split=0.1)
    
    validation_acc = np.amax(result.history['val_acc']) 
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multilayer perceptron")
    parser.add_argument("Path", help="path to terra workspace file")

    args = parser.parse_args()
    start = time.time()

    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=100,
                                          trials=Trials())

    print(args.Path)
    #X_train, Y_train, X_test, Y_test = data(args.Path)
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    
    print('\n')
    predictions = best_model.predict(X_test)
    predictions = [0 if x < 0.5 else 1 for x in predictions]
    print(confusion_matrix(Y_test, predictions))
    print('\n')
    print(classification_report(Y_test, predictions))
    print('\n')
    print(f'Accuracy score: {accuracy_score(Y_test, predictions)*100:.2f}%')
    print(f"Balanced accuracy score : {balanced_accuracy_score(Y_test, predictions)*100:.2f}%")

    end = time.time()
    print('completed in {} seconds'.format(end-start))

    