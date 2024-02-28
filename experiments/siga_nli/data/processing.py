from siga_nli.config import Config
import pandas as pd


def load_data(data_dir, first_split=0.8):

    dataset = pd.read_csv(data_dir)
    train_length = int(first_split * len(dataset))
    train_data = dataset.iloc[:train_length]
    validation_data = dataset.iloc[train_length:]
    return train_data, validation_data
