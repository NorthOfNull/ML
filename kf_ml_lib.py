#
# KF ML LIB
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

# For feature selection
from sklearn.ensemble import RandomForestClassifier

# Deep Learning Keras Imports
from keras import backend as K
from keras import models
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier



# Load dataset from file; returns a Pandas DataFrame
def load_dataset(path):
    dataset = pd.read_csv(path, index_col=0, low_memory=False)
    
    return dataset


# Split dataset into Feature Vector (X) and Label Vector (y) DataFrames
# Includes Feature Selection via RFC Classifier for the extended feature datasets
def split_dataset(dataset, extended, deep_learning=False):
    # Define feature_vector_columns for normal or extended feature vectors
    if extended:
        feature_vector_columns = ['sTos','dTos','SrcWin','DstWin','sHops','dHops','sTtl','dTtl','TcpRtt','SynAck','AckDat','SrcPkts','DstPkts','SrcBytes','DstBytes','SAppBytes','DAppBytes','Dur','TotPkts','TotBytes','TotAppByte','Rate','SrcRate','DstRate']
    else:
        feature_vector_columns = ['Dur','sTos','dTos','TotPkts','TotBytes','SrcBytes']
        
    # Defines the label_column_header
    label_vector_column = ['Label']


    # Seperate Dataframe into feature_vectors (X) and label_vector (y)
    X = dataset.loc[:, feature_vector_columns]
    y = dataset.loc[:, label_vector_column]
    y = np.ravel(y)

    # Feature Selection only applies to extended datasets in this case
    # Uses RFC to find the feature importances
    # Returns highest 15 features for the dataset. out of 24
    if extended and deep_learning is False:
        # Perform Feature Selection
        # Find feature importances via RFC feature importances attribute after fitting to dataset
        rfc = RandomForestClassifier()
        rfc = rfc.fit(X, y)


        # Associate each feature importance score with it's feature vector column name
        fi_f = zip(rfc.feature_importances_, feature_vector_columns)

        # Sort feature importances from high to low, maintaining the feature vector column name relationships
        fi_f = sorted(fi_f, key=lambda x: x[0], reverse=True)

        # Remove lowest 9 features, leaving 15 most important features
        fi_f = fi_f[:15]

        # Extract the feature selected feature columns
        selected_dataset_features = []

        for fi, f in fi_f:
            selected_dataset_features.append(f)

        del rfc, X

        # Return new selected_datset_features DataFrame with the selected features
        X = dataset.loc[:, selected_dataset_features]


    return X, y

    
# Counts normal and botnet flows from the labels within a given dataset
def get_flow_counts(dataset):
    normal_flows = 0
    botnet_flows = 0
    
    for row in range(dataset.shape[0]):    
        if dataset.at[row, 'Label'] == "Normal":
            normal_flows += 1
        if dataset.at[row, 'Label'] == "Botnet":
            botnet_flows += 1
    
    return normal_flows, botnet_flows


# Calculate's a classifier's F1-Score from Precision and Recall values
def calc_f1_score(precision, recall):
    f1_score = 2 * ((precision*recall) / (precision+recall))
    
    return f1_score


# Deep Learning Functions
#
#
# Create FFNN Model, taking in parameters for number of hidden layers and activation functions
# FFNN_1
def make_sequential_model(hidden_layers, layers_activation, output_activation, loss_function, optimiser_function):
    model = models.Sequential()

    model.add(Dense(hidden_layers, input_dim=24, activation=layers_activation))
    model.add(Dense(1, activation=output_activation))

    model.compile(loss=loss_function, optimizer=optimiser_function)

    return model


# Returns Keras Classifier for the Keras FFNN Model - For use in GA Bio-Optimisation
# Takes in parameters for the FFNN, and batch and epoch value
def build_keras_ffnn_classifier(epochs, batch_size, hidden_layers, layers_activation, output_activation, loss_function, optimiser_function):
    model = KerasClassifier(build_fn=make_sequential_model,
                                hidden_layers=hidden_layers,
                                layers_activation=layers_activation,
                                output_activation=output_activation,
                                loss_function=loss_function,
                                optimiser_function=optimiser_function,
                            epochs=epochs, batch_size=batch_size,
                            verbose=0
                            )

    return model


# FFNN_1
def make_sequential_model_2(hidden_layers, layers_activation, dropout_1, hidden_layers_2, input_dim_2, layers_activation_2, dropout_2, output_activation, loss_function, optimiser_function):
    model = models.Sequential()

    model.add(Dense(hidden_layers, input_dim=24, activation=layers_activation))
    model.add(Dropout(dropout_1))
    model.add(Dense(hidden_layers_2, input_dim=input_dim_2, activation=layers_activation_2))
    model.add(Dropout(dropout_2))
    model.add(Dense(1, activation=output_activation))

    model.compile(loss=loss_function, optimizer=optimiser_function)

    return model


# Returns Keras Classifier for the Keras FFNN_2 Model - For use in GA Bio-Optimisation
# Takes in parameters for the FFNN_2, and batch and epoch value
def build_keras_ffnn_classifier_2(epochs, batch_size, hidden_layers, layers_activation, dropout_1, hidden_layers_2, input_dim_2, layers_activation_2, dropout_2, output_activation, loss_function, optimiser_function):
    model = KerasClassifier(build_fn=make_sequential_model_2,
                                hidden_layers=hidden_layers,
                                layers_activation=layers_activation,
                                dropout_1=dropout_1,
                                hidden_layers_2=hidden_layers_2,
                                input_dim_2=input_dim_2,
                                layers_activation_2=layers_activation_2,
                                dropout_2=dropout_2,
                                output_activation=output_activation,
                                loss_function=loss_function,
                                optimiser_function=optimiser_function,
                            epochs=epochs, batch_size=batch_size,
                            verbose=0
                            )

    return model
