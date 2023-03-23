import pandas as pd
import numpy as np
from os.path import join
from nilmtk.datastore import Key
from nilmtk.measurement import LEVEL_NAMES
from nilmtk.utils import check_directory_exists, get_datastore, get_module_directory
from nilm_metadata import convert_yaml_to_hdf5
from copy import deepcopy

column_mapping = {
    'W': ('power', 'active'),
    'A': ('current', ''),
    'VA': ('power', 'apparent'),
    'VAR': ('power', 'reactive'),
}

TIMESTAMP_COLUMN_NAME = "timestamp"
TIMEZONE = "Europe/Berlin"
START_DATETIME, END_DATETIME = '2023-01-17', '2023-03-08'
FREQ = "10S"


def reindex_fill_na(df, idx):
    df_copy = deepcopy(df)
    df_copy = df_copy.reindex(idx)

    power_columns = [
        x for x in df.columns if x[0] in ['power']]
    non_power_columns = [x for x in df.columns if x not in power_columns]

    for power in power_columns:
        df_copy[power].fillna(0, inplace=True)
    for measurement in non_power_columns:
        df_copy[measurement].fillna(df[measurement].median(), inplace=True)

    return df_copy


def convert_dataset(data_path, output_filename, format="HDF"):
    """
    Parameters
    ----------
    data_path : str
        The root path of the dataset.
    output_filename : str
        The destination filename (including path and suffix).
    """

    check_directory_exists(data_path)
    idx = pd.date_range(start=START_DATETIME, end=END_DATETIME, freq=FREQ)
    idx = idx.tz_localize('UTC').tz_convert(TIMEZONE)

    # Open data store
    store = get_datastore(output_filename, format, mode='w')
    electricity_path = data_path

    # Mains data
    for chan in range(1, 7):
        key = Key(building=1, meter=chan)
        filename = join(electricity_path, "%d.csv" % chan)
        print('Loading ', chan)
        df = pd.read_csv(filename, dtype=np.float64, na_values='\\N')
        df.drop_duplicates(subset=["timestamp"], inplace=True)
        df.index = pd.to_datetime(df.timestamp.values, unit='s', utc=True)
        df = df.tz_convert(TIMEZONE)
        df = df.drop(TIMESTAMP_COLUMN_NAME, 1)
        df.columns = pd.MultiIndex.from_tuples(
            [column_mapping[x] for x in df.columns],
            names=LEVEL_NAMES
        )
        df = df.apply(pd.to_numeric, errors='ignore')
        df = df.dropna()
        df = df.astype(np.float32)
        df = df.sort_index()
        df = df.resample(FREQ).mean()
        df = reindex_fill_na(df, idx)
        assert df.isnull().sum().sum() == 0
        store.put(str(key), df)
    store.close()

    metadata_dir = 'metadata'
    print(metadata_dir)
    convert_yaml_to_hdf5(metadata_dir, output_filename)

    print("Done converting to HDF5!")


def main():
    convert_dataset('campus_juelich', 'NOWUM-Energy-Dataset.h5')


if __name__ == '__main__':
    main()
