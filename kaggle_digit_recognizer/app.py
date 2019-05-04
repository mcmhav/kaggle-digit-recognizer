from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd

import sys
import os
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from kaggle_digit_recognizer.data import read_data
from kaggle_digit_recognizer.preprocess import preprocess_data


def app():
    df_train, df_test = read_data()
    df_train_aligned, df_train_labels, df_test_aligned = preprocess_data(
        df_train, df_test)
