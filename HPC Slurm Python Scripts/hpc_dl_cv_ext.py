import kf_ml_lib as kf
import numpy as np

from sklearn.model_selection import cross_validate

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

# MPI Node
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print("Running on node ", rank)

scoring = ['precision_macro', 'recall_macro']

# Array of Classifiers under test
classifier = kf.build_keras_ffnn_classifier(5, 100, 100, 'relu', 'sigmoid', 'binary_crossentropy', 'adam') 

# Dataset to load
# Dataset_path_list is indexed by the node number that it is running on
dataset_path = kf.extended_dataset_path_list[rank]


# Load and Split Dataset
dataset = kf.load_dataset(dataset_path)
X, y = kf.split_dataset(dataset, extended=True, deep_learning=True)
del dataset

# Cross Validate the classifer on the full dataset
results = cross_validate(classifier, X, y, cv=10, scoring=scoring, n_jobs=10, verbose=0)

fit_time = np.mean(results['fit_time'])
precision = np.mean(results['test_precision_macro'])
recall = np.mean(results['test_recall_macro'])
f1_score = kf.calc_f1_score(precision, recall)

print(dataset_path)
print(classifier)
print("Fit Time = ", fit_time)
print("Precision = ", precision)
print("Recall = ", recall)
print("F1 Score = ", f1_score)
print("\n")
    
del X, y
print("////////////////////////////////////////////////////////////////////////\n\n")
