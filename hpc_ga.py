import kf_ml_lib as kf
import kf_bio_optimisation_lib as kf_bio

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

print("Running on node ", rank)

# DECISION TREE CLASSIFIER GA RESULTS
# Genetic Algorithm using 10 generations with a population size of 10 for each dataset
# Each chromosome is evaluated with Stratified 10-fold Cross Validation and 
# uses Heuristic Inital Population Generation
best_dataset_chromosomes = []

dataset_path = kf.dataset_path_list[rank]

ga = kf_bio.kf_genetic_algorithm(classifier='DecisionTreeClassifier', dataset_path=dataset_path,
                        population_size=20, generations=20, evaluation_method='cv', initial_pop_gen_method='heuristic',
                        extended_dataset=False)

sorted_score_chromosome_list = ga.run()

best_score_chromosome = sorted_score_chromosome_list[0]
print("Best score & chromosome for dataset ", dataset_path, " =", best_score_chromosome)

# Get the best result from GA for this dataset
best_dataset_chromosomes.append(best_score_chromosome)

del ga, sorted_score_chromosome_list, best_score_chromosome