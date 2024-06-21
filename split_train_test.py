"""
Script to split data in train, test & validation and save them for reuse
"""
import os
import pickle

import scipy
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def kl_divergence(df1, df2, name_1, name_2):
    """
    Calculate KL Divergence between class distribution of df1 & df2
    :param df1: dataframe 1
    :param df2: dataframe 2
    :param name_1: name of dataframe 1
    :param name_2: name of dataframe 2
    """
    dist_1 = np.unique(df1['category'], return_counts=True)[1] / df1.shape[0]
    dist_2 = np.unique(df2['category'], return_counts=True)[1] / df2.shape[0]
    print(f'KL Divergence between {name_1} and {name_2} : ', scipy.stats.entropy(dist_1, dist_2))


def main():
    # Read the given Dataframe
    dataset = pd.read_csv(os.path.join('data', 'pricerunner_aggregate.csv'))
    dataset = dataset.rename(
        {'Product ID': 'product_id', 'Product Title': 'product_title', ' Merchant ID': 'merchant_id',
         ' Category ID': 'category_id', ' Category Label': 'category_label', ' Cluster ID': 'cluster_id',
         ' Cluster Label': 'cluster_label'}, axis=1)

    # Encode the Category Label
    lb = LabelEncoder()
    lb.fit(dataset['category_label'])
    print(lb.classes_)
    # Save Class Name
    with open('model_files/class_names.pkl', 'wb') as f:
        pickle.dump(lb, f)

    dataset['category'] = lb.transform(dataset['category_label'])

    # Select Relevant Columns and Split dataset 80/20 into Train & Test
    train_df, test_df = train_test_split(dataset[['product_title', 'cluster_label', 'category']],
                                         stratify=dataset['category'], test_size=0.2)
    train_df, val_df = train_test_split(train_df,
                                        stratify=train_df['category'], test_size=0.25)
    print('No of Training Samples : ', train_df.shape[0])
    print('No of Validation Samples : ', val_df.shape[0])
    print('No of Test Samples : ', test_df.shape[0])

    # Check KL Divergence between train class distribution & test class distribution
    kl_divergence(train_df, val_df, 'train', 'val')
    kl_divergence(train_df, test_df, 'train', 'test')

    # Save Splitted Dataset
    train_df.to_csv(os.path.join('data', 'train_test', 'train_data.csv'), index=False)
    val_df.to_csv(os.path.join('data', 'train_test', 'val_data.csv'), index=False)
    test_df.to_csv(os.path.join('data', 'train_test', 'test_data.csv'), index=False)


if __name__ == '__main__':
    main()
