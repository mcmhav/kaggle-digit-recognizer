import pandas as pd


DATA_PATH = 'data'
TRAIN_DATA_SET = 'train.csv'
TEST_DATA_SET = 'test.csv'
SAMPLE_DATA_SET = 'sample_submission.csv'


def read_data():
    df_train = pd.read_csv(f'{DATA_PATH}/{TRAIN_DATA_SET}')
    df_test = pd.read_csv(f'{DATA_PATH}/{TEST_DATA_SET}')

    return df_train, df_test
