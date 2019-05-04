def preprocess_data(df_train, df_test):
    df_train_aligned = df_train.drop('label', axis=1) / 255
    df_train_labels = df_train['label']
    df_test_aligned = df_test / 255

    return df_train_aligned, df_train_labels, df_test_aligned


def preprocess_data_2d(df_train, df_test):
    Y_train = df_train["label"]
    X_train = df_train.drop(labels=["label"], axis=1)
    X_train = X_train.values.reshape(-1, 28, 28)

    X_test = df_test.values.reshape(-1, 28, 28)

    X_train = X_train / 255
    X_test = X_test / 255

    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    return X_train, Y_train, X_test
