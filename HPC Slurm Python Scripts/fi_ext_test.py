import kf_ml_lib as kf
import kf_bio_optimisation_lib as kf_bio
from sklearn.ensemble import RandomForestClassifier

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print("Running on node ", rank)


dataset_path = kf.extended_dataset_path_list[rank]
dataset = kf.load_dataset(dataset_path)
X, y = kf.split_dataset(dataset, extended=True)

rfc = RandomForestClassifier()
rfc = rfc.fit(X, y)

print("Dataset = ", rank)
print(rfc.feature_importances_)

feature_vector_columns = ['sTos','dTos','SrcWin','DstWin','sHops','dHops','sTtl','dTtl','TcpRtt','SynAck','AckDat','SrcPkts','DstPkts','SrcBytes','DstBytes','SAppBytes','DAppBytes','Dur','TotPkts','TotBytes','TotAppByte','Rate','SrcRate','DstRate']

# Associate each feature importance score with it's feature vector column name
fi_f = zip(rfc.feature_importances_, feature_vector_columns)


#print("Dataset = ", rank)
#print(fi_f)

# Sort feature importances from high to low, maintaining the feature vector column name relationships
fi_f = sorted(fi_f, key=lambda x: x[0], reverse=True)

#print("\nSorted:\n", fi_f)

# Remove lowest 9 features, leaving 15 features
fi_f = fi_f[:15]

#print("\nFeature removal:\n", fi_f)

#print(rfc.feature_importances_)
print("\n\n")

del dataset, X, y 

