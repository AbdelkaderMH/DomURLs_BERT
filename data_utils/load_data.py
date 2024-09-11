import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_dataset(path, label_column):

    le = LabelEncoder()

    df_train = pd.read_csv(os.path.join(path, 'train.csv'))
    df_dev = pd.read_csv(os.path.join(path, 'dev.csv'))
    df_test = pd.read_csv(os.path.join(path, 'test.csv'))

    df_train[label_column] = le.fit_transform(df_train[label_column])
    df_dev[label_column] = le.transform(df_dev[label_column])
    df_test[label_column] = le.transform(df_test[label_column])


    return {'train': df_train, 'dev': df_dev, 'test': df_test, 'label_encoder': le}