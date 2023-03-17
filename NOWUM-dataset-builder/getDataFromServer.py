import os
import numpy as np
import pandas as pd
import convert_to_hdf5_dataset
import yaml
from influxdb import DataFrameClient

unit_names = {
    'P_ges': 'W',
    'Q_ges': 'VAR',
    'S_ges': 'VA',
    'Power': 'W',
    'ApparentPower': 'VA',
    'ReactivePower': 'VAR'
}


def getdata():
    # read config.yaml file to get connection details
    with open('config.yaml', 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    device_data = []
    for key in config.keys():
        influx_con = DataFrameClient(host=config[key].get('influxHost'),
                                     port=config[key].get("influxPort"),
                                     username=config[key].get('influxUsername'),
                                     password=config[key].get('influxPassword'),
                                     database=config[key].get('influxDatabase'),
                                     timeout=10)
        devices = config[key].get('devices')
        for device in devices:
            device_data.append([device.get('type'), data_per_device(influx_con,
                                                                    config[key].get('tag'),
                                                                    config[key].get('valueNames'),
                                                                    config[key].get('measurement'),
                                                                    device['name'])])

    return device_data


def data_per_device(influx_conn: DataFrameClient, tag: str, value_names: str, measurement: str, device_name: str) -> pd.DataFrame:
    sql_request = f'SELECT {value_names} FROM {measurement} WHERE {tag} = \'{device_name}\''
    request = influx_conn.query(sql_request)
    print(request)
    request = request[measurement]
    # convert datetime to unix timestamp
    request.index = pd.DatetimeIndex(request.index).astype(np.int64) / 1e9
    # rename columns without inplace=True
    request = request.rename(columns=unit_names)
    return request


def main():
    data = getdata()
    i = 1
    # check if labels.dat exists and delete it
    try:
        os.remove('campus_juelich/labels.dat')
    except OSError:
        pass
    for device in data:
        device[1].to_csv(f'campus_juelich/{i}.csv', index_label='timestamp')
        # save to campus_juelich/labels.dat
        with open('campus_juelich/labels.dat', 'a') as f:
            f.write(f'{i} {device[0]}\n')
        i += 1

    print('Done getting data from server! Now converting to HDF5...')

    convert_to_hdf5_dataset.main()


if __name__ == '__main__':
    main()
