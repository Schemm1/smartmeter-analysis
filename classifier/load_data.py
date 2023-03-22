import numpy as np
import pandas as pd
from nilmtk import DataSet
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Flatten, Convolution1D, LSTM, ConvLSTM2D, Reshape, Convolution2D
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.callbacks
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import model_from_json
import pickle
import json

# dict with dataset and building number
data = {
    # '../NOWUM-dataset-builder/NOWUM-Energy-Dataset.h5':
    #     {
    #         'buildings': [1],
    #         'start_time': None,
    #         'end_time': None
    #     }
    '../ukdale.h5':
        {
            'buildings': [1, 5],
            'start_time': None,
            'end_time': None
            # 'start_time': '2013-04-01',
            # 'end_time': '2013-05-01'
        }
}


def create_dataset(data: dict, appliances: list) -> (np.array, np.array):
    """
    Create a dataset from the data dict and the appliances list.

    Parameters
    ----------
    data : dict
        The root path of the dataset and the buildings that should be used.
    appliances : list
        The list of appliances to be used.

    Returns
    -------
    x : numpy array
        The features
    y : numpy array
        The labels
    """

    # x and y as numpy arrays where x_temp and y_temp get appended
    x = np.empty((0, 2))
    y = np.empty((0, len(appliances)))

    for path in data.keys():
        dataset = DataSet(path)
        print(f'Loading data from {path}')
        buildings = data[path].get('buildings')
        start_time = data[path].get('start_time')
        end_time = data[path].get('end_time')
        if start_time is not None and end_time is not None:
            dataset.set_window(start=start_time, end=end_time)
        for building in buildings:
            print(f'Loading data from building {building}')
            x_temp = pd.DataFrame(next(dataset.buildings[building].elec.mains().load()))
            x_temp.columns = x_temp.columns.droplevel()
            # only keep the columns 'apparent' and 'active'
            x_temp = x_temp[['apparent', 'active']]
            # Y contains the activations for each appliance
            # When there is no activation, the value is 0
            for appliance in appliances:
                activations = dataset.buildings[building].elec.submeters().select_using_appliances(
                    type=appliance).get_activations()
                activations = pd.concat(activations)
                activations = pd.DataFrame(activations)
                activations.columns = [appliance]
                x_temp = x_temp.join(activations, how='outer').fillna(0)
            y_temp = x_temp.drop(columns=['apparent', 'active'])
            x_temp = x_temp[['apparent', 'active']]
            # resample X and Y to 10 seconds
            x_temp = x_temp.resample('10S').mean()
            y_temp = y_temp.resample('10S').mean()
            # drop all rows with nan values
            x_temp = x_temp.dropna()
            y_temp = y_temp.dropna()
            # append to x and y
            x = np.append(x, x_temp.values, axis=0)
            y = np.append(y, y_temp.values, axis=0)

    return x, y


def reduce_zeros(x: np.array, y: np.array) -> (np.array, np.array):
    """
    Reduce the number of 0s in the dataset.
    Parameters
    ----------
    x : numpy array
        The features
    y : numpy array
        The labels
    Returns
    -------
    x : numpy array
        The features
    y : numpy array
        The labels
    """
    # get the indices of the rows that contain only 0s
    indices = np.where(~y.any(axis=1))[0]
    # delete randomly 75 % of the rows that contain only 0s
    indices = np.random.choice(indices, int(len(indices) / 2), replace=False)
    x = np.delete(x, indices, axis=0)
    y = np.delete(y, indices, axis=0)
    return x, y


def scale_data(x: np.array, y: np.array) -> (np.array, np.array):
    """
    Scale the data using MinMaxScaler.

    Parameters
    ----------
    x : numpy array
        The data to be scaled.
    y : numpy array
        The labels which will be set to 0 or 1 if there is an activation.
    Returns
    -------
    x : numpy array
        The scaled data.
    """
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    # y will be set to 1 if the value is greater than 5 and 0 otherwise
    y = np.where(y > 5, 1, 0)
    return x, y


def create_windowed_data(x: np.array, y: np.array, window_size: int) -> (np.array, np.array):
    """
    Create windows from the data and labels.
    Each window contains window_size samples.

    Parameters
    ----------
    x : numpy array
        Features
    y : numpy array
        Labels
    window_size : int
        The size of the window.

    Returns
    -------
    x : numpy array
        Feature data containing windows of size window_size.
    y : numpy array
        Label data containing 0 or 1 if there is an activation in the window for each appliance.
    """
    x = np.array([x[i:i + window_size] for i in range(len(x) - window_size)])
    # y = np.array([y[i + window_size] for i in range(len(y) - window_size)])
    # y is Label data containing 0 or 1 if there is an activation in the window for each appliance
    y = np.array([y[i:i + window_size] for i in range(len(y) - window_size)])
    # y_ contains one row and as many columns as y has
    # if one row of y contains a 1, the corresponding column of y_ will be 1
    y_ = np.zeros((len(y), len(y[0][0])))
    for i in range(len(y)):
        for j in range(len(y[0][0])):
            if 1 in y[i, :, j]:
                y_[i, j] = 1

    # y = np.array([1 if 1 in y[:, i] else 0 for i in range(len(y[0]))])

    # y = np.array([1 if 1 in y[i:i + window_size] else 0 for i in range(len(y) - window_size)])
    return x, y_


def shuffle_data(x: np.array, y: np.array) -> (np.array, np.array):
    """
    Shuffle the data and labels.
    This is done to prevent the model from learning the order of the data.
    Also, long sequences of the same label can be avoided.

    Parameters
    ----------
    x : numpy array
        Features to be shuffled.
    y : numpy array
        Labels to be shuffled.

    Returns
    -------
    x : numpy array
        The shuffled features.
    y : numpy array
        The shuffled labels.
    """
    p = np.random.permutation(len(x))
    return x[p], y[p]


def createcnn_model(x_train, y_train, epochs, window, features):
    """
    Create the CNN model.
    Parameters
    ----------
    x_train : numpy array
        The training data.
    y_train : numpy array
        The training labels.
    epochs : int
        The number of epochs.
    window : int
        The size of the window.
    features : int
        The number of features.
    Returns
    -------
    model : keras model
        The CNN model.
    history : keras history
        The history of the training.
    """

    output_architecture = './tmpdata/convnet_architecture.json'
    best_weights_during_run = './tmpdata/weights.h5'
    final_weights = './tmpdata/weights.h5'
    loss_history = './tmpdata/history.pickle'

    class LossHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.train_losses = []
            self.valid_losses = []

        def on_epoch_end(self, epoch, logs={}):
            self.train_losses.append(logs.get('loss'))
            self.valid_losses.append(logs.get('val_loss'))

    model = Sequential()
    # the input shape is (window, features)
    model.add(Input(shape=(window, features,)))
    # reshape the input to (window, features, 1)
    model.add(Reshape((window, features, 1)))
    model.add(Convolution1D(filters=16, kernel_size=4, padding="valid", activation="relu"))
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(4, activation="sigmoid"))

    model.compile(optimizer='adam', loss='binary_crossentropy')
    history = LossHistory()
    checkpointer = ModelCheckpoint(filepath=best_weights_during_run, save_best_only=True, verbose=1)
    print('\n now training the model ... \n')
    model.fit(x_train, y_train, epochs=epochs, verbose=1, shuffle=True, callbacks=[history, checkpointer],
              validation_split=0.2, steps_per_epoch=3000)

    losses_dic = {'train_loss': history.train_losses, 'valid_loss': history.valid_losses}

    with open(loss_history, 'wb') as handle:
        pickle.dump(losses_dic, handle)

    print('\n saving the architecture of the model \n')
    json_string = model.to_json()
    open(output_architecture, 'w').write(json_string)

    print('\n saving the final weights ... \n')
    model.save_weights(final_weights, overwrite=True)
    print('done saving the weights')

    print('\n saving the training and validation losses')

    print('This was the model trained')
    print(model.summary())


def main():
    x_train, y_train = create_dataset(data, ['television', 'microwave', 'kettle', 'coffee maker'])
    print(f'x shape: {x_train.shape}')
    print(f'y shape: {y_train.shape}')
    x_train, y_train = reduce_zeros(x_train, y_train)
    print(f'x shape after reducing zeros: {x_train.shape}')
    print(f'y shape after reducing zeros: {y_train.shape}')
    x_train, y_train = scale_data(x_train, y_train)
    x_train, y_train = create_windowed_data(x_train, y_train, 6)
    x_train, y_train = shuffle_data(x_train, y_train)
    createcnn_model(x_train, y_train, 25, 6, 2)


if __name__ == '__main__':
    main()
