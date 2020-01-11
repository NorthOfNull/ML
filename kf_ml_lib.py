#
# KF ML LIB
#     TODO:
#          - TEST LOADING AND SPLITTING THE EXTENDED DATASET 1.CSV
#
#

dataset_path_list = ["../Datasets/CTU-13/Pre-processed/1.csv",
                    "../Datasets/CTU-13/Pre-processed/2.csv",
                    "../Datasets/CTU-13/Pre-processed/3.csv",
                    "../Datasets/CTU-13/Pre-processed/4.csv",
                    "../Datasets/CTU-13/Pre-processed/5.csv",
                    "../Datasets/CTU-13/Pre-processed/6.csv",
                    "../Datasets/CTU-13/Pre-processed/7.csv",
                    "../Datasets/CTU-13/Pre-processed/8.csv",
                    "../Datasets/CTU-13/Pre-processed/9.csv",
                    "../Datasets/CTU-13/Pre-processed/10.csv",
                    "../Datasets/CTU-13/Pre-processed/11.csv",
                    "../Datasets/CTU-13/Pre-processed/12.csv",
                    "../Datasets/CTU-13/Pre-processed/13.csv",
                    "../Datasets/ISOT Botnet 2010/Pre-processed/isot_botnet.csv",
                    "../Datasets/ISCX Botnet 2014/Pre-processed/iscx_training.csv"
                    ]

extended_dataset_path_list = ["../Datasets/CTU-13/Pre-processed_Extended/1.csv",
                            "../Datasets/CTU-13/Pre-processed_Extended/2.csv",
                            "../Datasets/CTU-13/Pre-processed_Extended/3.csv",
                            "../Datasets/CTU-13/Pre-processed_Extended/4.csv",
                            "../Datasets/CTU-13/Pre-processed_Extended/5.csv",
                            "../Datasets/CTU-13/Pre-processed_Extended/6.csv",
                            "../Datasets/CTU-13/Pre-processed_Extended/7.csv",
                            "../Datasets/CTU-13/Pre-processed_Extended/8.csv",
                            "../Datasets/CTU-13/Pre-processed_Extended/9.csv",
                            "../Datasets/CTU-13/Pre-processed_Extended/10.csv",
                            "../Datasets/CTU-13/Pre-processed_Extended/11.csv",
                            "../Datasets/CTU-13/Pre-processed_Extended/12.csv",
                            "../Datasets/CTU-13/Pre-processed_Extended/13.csv",
                            "../Datasets/ISOT Botnet 2010/Pre-processed_Extended/isot_botnet.csv",
                            "../Datasets/ISCX Botnet 2014/Pre-processed_Extended/iscx_training.csv"
                            ]

# Imports
import pandas as pd
import numpy as np


# Load dataset from file; returns a Pandas DataFrame
def load_dataset(path):
    dataset = pd.read_csv(path, index_col=0)
    
    return dataset


# Split dataset into Feature Vector and Label Vector DataFrames
def split_dataset(dataset, extended):
    # Define feature_vector_columns for normal or extended feature vectors
    if extended:
        feature_vector_columns = ['sTos','dTos','SrcWin','DstWin','sHops','dHops','sTtl','dTtl','TcpRtt','SynAck','AckDat','SrcPkts','DstPkts','SrcBytes','DstBytes','SAppBytes','DAppBytes','Dur','TotPkts','TotBytes','TotAppByte','Rate','SrcRate','DstRate']
    else:
        feature_vector_columns = ['Dur','sTos','dTos','TotPkts','TotBytes','SrcBytes']
        
    # Defines the label_column_header
    label_vector_column = ['Label']

    # Seperate Dataframe into feature_vectors from label_vector
    feature_vectors = dataset.loc[:, feature_vector_columns]
    label_vector = dataset.loc[:, label_vector_column]
        
    # Convert label_vector to 1D Array
    label_vector = np.ravel(label_vector)

    return feature_vectors, label_vector

    
# Counts normal and botnet flows from the labels within a given dataset
def get_flows(dataset):
    normal_flows = 0
    botnet_flows = 0
    
    for row in range(dataset.shape[0]):    
        if dataset.at[row, 'Label'] == "Normal":
            normal_flows += 1
        if dataset.at[row, 'Label'] == "Botnet":
            botnet_flows += 1

    print('Normal flows = ', normal_flows)
    print('Botnet flows = ', botnet_flows)
    
    return normal_flows, botnet_flows


# Calculate's a classifier's F1-Score from Precision and Recall values
def calc_f1_score(precision, recall):
    f1_score = 2 * ((precision*recall) / (precision+recall))
    
    return f1_score


