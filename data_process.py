"""Data preprocessing."""

import os
import pickle
from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_pickle(f: str) -> Any:
    """Load a pickle file.

    Parameters:
        f: the pickle filename

    Returns:
        the pickled data
    """
    return pickle.load(f, encoding="latin1")


def get_MUSHROOM_data(validation: float, testing: float = 0.2) -> dict:
    """Load the mushroom dataset.

    Parameters:
        validation: portion of the dataset used for validation
        testing: portion of the dataset used for testing

    Returns
        the train/val/test data and labels
    """
    X_train = np.load("mushroom/X_train.npy")
    y_train = np.load("mushroom/y_train.npy")
    y_test = np.load("mushroom/y_test.npy")
    X_test = np.load("mushroom/X_test.npy")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=validation / (1 - testing), random_state=123
    )
    data = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }
    return data


def construct_MUSHROOM():
    """Convert raw categorical data from mushroom dataset to one-hot encodings.
    """
    dataset = pd.read_csv("mushroom/mushrooms.csv")
    y = dataset["class"]
    X = dataset.drop("class", axis=1)
    Encoder_X = LabelEncoder()
    for col in X.columns:
        X[col] = Encoder_X.fit_transform(X[col])
    Encoder_y = LabelEncoder()
    y = Encoder_y.fit_transform(y)
    X = X.values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )
    np.save("mushroom/X_train.npy", X_train)
    np.save("mushroom/y_train.npy", y_train)
    np.save("mushroom/X_test.npy", X_test)
    np.save("mushroom/y_test.npy", y_test)
