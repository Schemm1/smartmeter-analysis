import numpy as np
import pandas as pd
import nilmtk
from nilmtk import DataSet
from sklearn.preprocessing import MinMaxScaler

# dict with dataset and building number
data = {
    '../NOWUM-dataset-builder/NOWUM-Energy-Dataset.h5':
        {
            'buildings': [1],
            'start_time': None,
            'end_time': None
        },
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

            # append to x and y
            x = np.append(x, x_temp.values, axis=0)
            y = np.append(y, y_temp.values, axis=0)

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


def main():
    # x, y = get_traing_data(data, ['television', 'microwave', 'electric shower heater', 'kettle', 'coffee maker'])
    x_train, y_train = create_dataset(data, ['television', 'microwave', 'kettle', 'coffee maker'])
    x_train_scaled, y_train_scaled = scale_data(x_train, y_train)


if __name__ == '__main__':
    main()
