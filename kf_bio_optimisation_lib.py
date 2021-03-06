import random
import numpy as np
import kf_ml_lib as kf
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC


# Genetic Algorithm Class for hyperparamter optimisation
#          generate_inital_population()
#          run():
#               evaluate_fitness()
#               selection()
#               crossover()
#               generate_random_chromosome()
#               mutate_population()
#          plot_fitness_scores()
class kf_genetic_algorithm:
    def __init__(self, classifier, dataset_path, population_size, generations, evaluation_method, initial_pop_gen_method, extended_dataset):
        self.classifier = classifier
        self.population_size = population_size
        self.generations = generations
        self.evaluation_method = evaluation_method
        self.initial_pop_gen_method = initial_pop_gen_method
        self.extended_dataset = extended_dataset
        
        self.population = []
        self.best_generation_fitness_scores = []
        self.best_generation_chromosomes = []
        
        # Gene pool
        # Set options and constraints for the selected classifier's hyperparameters
        if self.classifier == "DecisionTreeClassifier":
            self.criterions = ['gini', 'entropy']
            self.splits = ['best', 'random']
            self.min_samples_splits = dict(low = 2, high = 5)
            self.min_samples_leafs = dict(low = 1, high = 4)
            self.min_weight_fraction_leafs = dict(low = 0.0, high = 0.1)
            self.class_weights = ['balanced', None]
        elif self.classifier == 'RandomForestClassifier':
            self.n_estimators = dict(low = 10, high = 200)
            self.criterions = ['gini', 'entropy']
            self.min_samples_splits = dict(low = 2, high = 5)
            self.min_samples_leafs = dict(low = 1, high = 4)
            self.min_weight_fraction_leafs = dict(low = 0.0, high = 0.1)
            self.class_weights = ['balanced', None]
        elif self.classifier == 'KNeighborsClassifier':
            self.n_neighbors = dict(low = 1, high = 10)
            self.weights = ['uniform', 'distance']
            self.algorithm = ['ball_tree', 'kd_tree']
            self.leaf_size = dict(low = 1, high = 50)
            self.p = dict(low = 1, high = 5)
        elif self.classifier == 'AdaBoostClassifier':
            self.n_estimators = dict(low = 5, high = 100)
            self.learning_rate = dict(low = 0.1, high = 1.0)
            self.algorithm = ['SAMME', 'SAMME.R']
            self.random_state = dict(low = 1, high = 50)
        elif self.classifier == 'LinearSVC':
            self.loss = ['hinge', 'squared_hinge']
            self.tol = dict(low = 1e-5, high = 0.1)
            self.c = dict(low = 1, high = 5)
        elif self.classifier == 'FFNN':
            self.epochs = dict(low = 2, high = 10)
            self.batch_size = dict(low = 10, high = 1000)
            self.layer_units = dict(low = 10, high = 1000)
            self.layers_activation = ['softmax', 'selu', 'softplus', 'relu', 'tanh', 'sigmoid', 'linear']
            self.output_activation = ['softmax', 'selu', 'softplus', 'relu', 'tanh', 'sigmoid', 'linear']
            self.loss_function = ['mean_squared_error', 'mean_absolute_error', 'squared_hinge', 'binary_crossentropy']
            self.optimiser_function = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
        elif self.classifier == 'FFNN_2':
            self.epochs = dict(low = 2, high = 10)
            self.batch_size = dict(low = 10, high = 1000)
            self.layer_units = dict(low = 10, high = 1000)
            self.layers_activation = ['softmax', 'selu', 'softplus', 'relu', 'tanh', 'sigmoid', 'linear']
            self.dropout_1 = dict(low = 0.1, high = 0.5)
            self.layer_units_2 = dict(low = 10, high = 1000)
            self.input_dim_2 = dict(low = 2, high = 24)
            self.layers_activation_2 = ['softmax', 'selu', 'softplus', 'relu', 'tanh', 'sigmoid', 'linear']
            self.dropout_2 = dict(low = 0.1, high = 0.5)
            self.output_activation = ['softmax', 'selu', 'softplus', 'relu', 'tanh', 'sigmoid', 'linear']
            self.loss_function = ['mean_squared_error', 'mean_absolute_error', 'squared_hinge', 'binary_crossentropy']
            self.optimiser_function = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'] 
        else:
            raise Exception("Classifier not supported: ", self.classifier)
                
        
        if self.initial_pop_gen_method not in ['heuristic', 'random']:
            raise Exception("Initial Population Generation Method not supported / defined ", self.initial_pop_gen_method)
            
        # Generate inital chromosome population, with size population_size
        self.population = self.generate_inital_population()
        
        
        # Load, process and split dataset
        # Dataset split includes feature selection, IF the extened datasets are to be used
        if self.extended_dataset not in [True, False]:
            raise Exception("Dataset Format not defined", self.extended_dataset)

        self.dataset = kf.load_dataset(dataset_path)

        # If Deep Learning is being used (FFNN or FFNN_2 classifiers), then we define the 
        # dataset splitting to be for deep_learning, whereby feature selection does not take place
        if self.classifier in ['FFNN', 'FFNN_2']:
            self.X, self.y = kf.split_dataset(self.dataset, extended=self.extended_dataset, deep_learning=True)
        else:
            self.X, self.y = kf.split_dataset(self.dataset, extended=self.extended_dataset)
        
        # Evaluation method 
        if self.evaluation_method == 'single_fit_eval':
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size=0.7, stratify=self.y)
            del self.dataset
        elif self.evaluation_method == 'cv':
            del self.dataset
            #print("\nEvaluation Method = Cross Validation\nThis may take a while...")
        else:
            raise Exception("Evalutaion Method not supported:", self.evaluation_method)
            
        
    def __del__(self):
        print("\nDestroying ga object\n")

        
    # Function to generate an inital random population of chromosomes, given a population_size
    def generate_inital_population(self):
        # Inital chromosome population
        population = []
        start = 0
        
        # First chromosome with default parameters, if initial_pop_gen_method=='heuristic'
        if self.initial_pop_gen_method == 'heuristic':
            start = 1
            chromosome = []
            
            # SKLearn Classifiers 
            if self.classifier == 'DecisionTreeClassifier':
                chromosome.append('gini')
                chromosome.append('best')
                chromosome.append(2)
                chromosome.append(1)
                chromosome.append(0.0)
                chromosome.append(None)
            elif self.classifier == 'RandomForestClassifier':
                chromosome.append(100)
                chromosome.append('gini')
                chromosome.append(2)
                chromosome.append(1)
                chromosome.append(0.0)
                chromosome.append(None)
            elif self.classifier == 'KNeighborsClassifier':
                chromosome.append(5)
                chromosome.append('uniform')
                chromosome.append('kd_tree')
                chromosome.append(30)
                chromosome.append(2)
            elif self.classifier == 'AdaBoostClassifier':
                chromosome.append(50)
                chromosome.append(1.0)
                chromosome.append('SAMME.R')
                chromosome.append(None)
            elif self.classifier == 'LinearSVC':
                chromosome.append('squared_hinge')
                chromosome.append(1e-4)
                chromosome.append(1)
            # Keras Neural Network Classifier
            elif self.classifier == 'FFNN':
                chromosome.append(6)
                chromosome.append(100)
                chromosome.append(500)
                chromosome.append('relu')
                chromosome.append('sigmoid')
                chromosome.append('binary_crossentropy')
                chromosome.append('adam')
            elif self.classifier == 'FFNN_2':
                chromosome.append(6)
                chromosome.append(100)
                chromosome.append(500)
                chromosome.append('relu')
                chromosome.append(0.25)
                chromosome.append(500)
                chromosome.append(18)
                chromosome.append('relu')
                chromosome.append(0.25)
                chromosome.append('sigmoid')
                chromosome.append('binary_crossentropy')
                chromosome.append('adam')
                
            # Append the injected chromosome to the population
            population.append(chromosome)
        
        # Fill each chromosome with randomly selected genes from the classifier's gene pools 
        # and add it to the population
        for i in range(start, self.population_size):
            chromosome = self.generate_random_chromosome()
            
            population.append(chromosome)
        
        return population

    
    # Run the Genetic Algorithm for N generations
    def run(self):        
        for generation in range(0, self.generations):
            # Generate fitness scores for all the chromosomes in the inital population
            fitness_scores = self.evaluate_fitness()   
            
            # Select the best two performing chromosomes to be parents
            parent1, parent2 = self.selection(fitness_scores)

            # Mate parents using one-point-crossover in order to generate offspring
            offspring1, offspring2 = self.crossover(parent1, parent2)
            
            # Fill new population with 25% offspring1 and offspring2
            # and generate random chromosomes for the remainder
            self.population = []

            for index in range(0, self.population_size):
                #if index < self.population_size/2:
                if index < self.population_size/4:
                    chromosome = offspring1.copy()
                elif index < self.population_size/2:
                    chromosome = offspring2.copy()
                else:
                    chromosome = self.generate_random_chromosome()
                    
                self.population.append(chromosome)
            
            # Mutate the population
            self.mutate_population()     
        
        sorted_best_results = sorted(zip(self.best_generation_fitness_scores, self.best_generation_chromosomes), key=lambda x: x[0], reverse=True)
        
        return sorted_best_results
        
    
    # Generate fitness_scores for the given population
    def evaluate_fitness(self):
        fitness_scores = []
        
        for chromosome in self.population:
            # Setup classifier with the chromosome
            if self.classifier == 'DecisionTreeClassifier':
                clf = DecisionTreeClassifier(criterion=chromosome[0],
                                             splitter=chromosome[1],
                                             min_samples_split=chromosome[2],
                                             min_samples_leaf=chromosome[3],
                                             min_weight_fraction_leaf=chromosome[4],
                                             class_weight=chromosome[5])
            elif self.classifier == 'RandomForestClassifier':
                clf = RandomForestClassifier(n_jobs=1,
                                             n_estimators=chromosome[0],
                                             criterion=chromosome[1],
                                             min_samples_split=chromosome[2],
                                             min_samples_leaf=chromosome[3],
                                             min_weight_fraction_leaf=chromosome[4],
                                             class_weight=chromosome[5])
            elif self.classifier == 'KNeighborsClassifier':
                clf = KNeighborsClassifier(n_jobs=1,
                                           n_neighbors=chromosome[0],
                                           weights=chromosome[1],
                                           algorithm=chromosome[2],
                                           leaf_size=chromosome[3],
                                           p=chromosome[4])
            elif self.classifier == 'AdaBoostClassifier':
                clf = AdaBoostClassifier(n_estimators=chromosome[0],
                                         learning_rate=chromosome[1],
                                         algorithm=chromosome[2],
                                         random_state=chromosome[3])
            elif self.classifier == 'LinearSVC':
                clf = LinearSVC(loss=chromosome[0],
                                tol=chromosome[1],
                                C=chromosome[2])
            # Keras Deep Learning FFNN Classifier:
            elif self.classifier == 'FFNN':
                clf = kf.build_keras_ffnn_classifier(epochs=chromosome[0],
                                                     batch_size=chromosome[1],
                                                     layer_units=chromosome[2],
                                                     layers_activation=chromosome[3],
                                                     output_activation=chromosome[4],
                                                     loss_function=chromosome[5],
                                                     optimiser_function=chromosome[6])
            elif self.classifier == 'FFNN_2':
                clf = kf.build_keras_ffnn_classifier_2(epochs=chromosome[0],
                                                     batch_size=chromosome[1],
                                                     layer_units=chromosome[2],
                                                     layers_activation=chromosome[3],
                                                     dropout_1=chromosome[4],
                                                     layer_units_2=chromosome[5],
                                                     input_dim_2=chromosome[6],
                                                     layers_activation_2=chromosome[7],
                                                     dropout_2=chromosome[8],
                                                     output_activation=chromosome[9],
                                                     loss_function=chromosome[10],
                                                     optimiser_function=chromosome[11])


                
            # Evaluate fitness, depending on the evaluation method requested
            if self.evaluation_method == 'single_fit_eval':
                clf = clf.fit(self.X_train, self.y_train)
            
                # Make predictions on test data for this chromosome
                predictions = clf.predict(self.X_test)
            
                # Evaluate model Precision, Recall and F1-Score performance
                precision = metrics.precision_score(self.y_test, predictions, zero_division=0, pos_label='Botnet')
                recall = metrics.recall_score(self.y_test, predictions, pos_label='Botnet')
            
                if precision == 0:
                    f1_score = 0
                else:
                    f1_score = kf.calc_f1_score(precision, recall)

            elif self.evaluation_method == 'cv':
                scoring = ['precision_macro', 'recall_macro']
                
                results = cross_validate(clf, self.X, self.y, cv=10, scoring=scoring, n_jobs=10, verbose=0)

                fit_time = np.mean(results['fit_time'])
                precision = np.mean(results['test_precision_macro'])
                recall = np.mean(results['test_recall_macro'])
                
                if precision == 0:
                    f1_score = 0
                else:
                    f1_score = kf.calc_f1_score(precision, recall)

            fitness_scores.append(f1_score)
    
        return fitness_scores
    
    
    # Perform tournament selection on the evaluated chromosomes; returning the two best performers
    def selection(self, fitness_scores):        
        # Sort, from highest to lowest, the fitness_scores and their chromosomes
        sorted_results = sorted(zip(fitness_scores, self.population), key=lambda x: x[0], reverse=True)
        
        # Remove 'nan' fitness_value results from the sorted_results
        for result in sorted_results:
            if np.isnan(result[0]):
                sorted_results.remove(result)
        
        # Get two fittest chromosomes from the sorted chromosomes
        parent1 = sorted_results[0]
        parent2 = sorted_results[1]
        
        self.best_generation_fitness_scores.append(parent1[0])
        self.best_generation_chromosomes.append(parent1[1])
        
        # Remove fitness scores from chromosome
        parent1 = parent1[1]
        parent2 = parent2[1]

        return parent1, parent2
        
        
    # Performs one-point-crossover of the parents, returning the two offspring
    def crossover(self, parent1, parent2):
        chromosome_length = len(parent1)
        offspring1 = []
        offspring2 = []

        for gene in range(chromosome_length):
            if(gene < (chromosome_length / 2)):
                offspring1.append(parent1[gene])
                offspring2.append(parent2[gene])
            else:
                offspring1.append(parent2[gene])
                offspring2.append(parent1[gene])

        return offspring1, offspring2
    
    
    # Generate a random chromosome for the defined classifier
    def generate_random_chromosome(self):
        chromosome = []
        
        if self.classifier == 'DecisionTreeClassifier':
            chromosome.append(random.choice(self.criterions))
            chromosome.append(random.choice(self.splits))
            chromosome.append(random.randrange(self.min_samples_splits['low'], self.min_samples_splits['high']))
            chromosome.append(random.randrange(self.min_samples_leafs['low'], self.min_samples_leafs['high']))
            chromosome.append(random.uniform(self.min_weight_fraction_leafs['low'], self.min_weight_fraction_leafs['high']))  
            chromosome.append(random.choice(self.class_weights))
        elif self.classifier == 'RandomForestClassifier':
            chromosome.append(random.randrange(self.n_estimators['low'], self.n_estimators['high']))
            chromosome.append(random.choice(self.criterions))
            chromosome.append(random.randrange(self.min_samples_splits['low'], self.min_samples_splits['high']))
            chromosome.append(random.randrange(self.min_samples_leafs['low'], self.min_samples_leafs['high']))
            chromosome.append(random.uniform(self.min_weight_fraction_leafs['low'], self.min_weight_fraction_leafs['high']))  
            chromosome.append(random.choice(self.class_weights))
        elif self.classifier == 'KNeighborsClassifier':
            chromosome.append(random.randrange(self.n_neighbors['low'], self.n_neighbors['high']))
            chromosome.append(random.choice(self.weights))
            chromosome.append(random.choice(self.algorithm))
            chromosome.append(random.randrange(self.leaf_size['low'], self.leaf_size['high']))
            chromosome.append(random.randrange(self.p['low'], self.p['high']))
        elif self.classifier == 'AdaBoostClassifier':
            chromosome.append(random.randrange(self.n_estimators['low'], self.n_estimators['high']))
            chromosome.append(random.uniform(self.learning_rate['low'], self.learning_rate['high']))
            chromosome.append(random.choice(self.algorithm))
            chromosome.append(random.randrange(self.random_state['low'], self.random_state['high']))
        elif self.classifier == 'LinearSVC':
            chromosome.append(random.choice(self.loss))
            chromosome.append(random.uniform(self.tol['low'], self.tol['high']))
            chromosome.append(random.randrange(self.c['low'], self.c['high']))

        # Keras Deep Learning FFNN Classifier
        elif self.classifier == 'FFNN': 
            chromosome.append(random.randrange(self.epochs['low'], self.epochs['high'])) 
            chromosome.append(random.randrange(self.batch_size['low'], self.batch_size['high']))
            chromosome.append(random.randrange(self.layer_units['low'], self.layer_units['high']))
            chromosome.append(random.choice(self.layers_activation))
            chromosome.append(random.choice(self.output_activation))
            chromosome.append(random.choice(self.loss_function))
            chromosome.append(random.choice(self.optimiser_function))
        elif self.classifier == 'FFNN_2': 
            chromosome.append(random.randrange(self.epochs['low'], self.epochs['high'])) 
            chromosome.append(random.randrange(self.batch_size['low'], self.batch_size['high']))
            chromosome.append(random.randrange(self.layer_units['low'], self.layer_units['high']))
            chromosome.append(random.choice(self.layers_activation))
            chromosome.append(random.uniform(self.dropout_1['low'], self.dropout_1['high']))
            chromosome.append(random.randrange(self.layer_units_2['low'], self.layer_units_2['high']))
            chromosome.append(random.randrange(self.input_dim_2['low'], self.input_dim_2['high']))
            chromosome.append(random.choice(self.layers_activation_2))
            chromosome.append(random.uniform(self.dropout_2['low'], self.dropout_2['high']))
            chromosome.append(random.choice(self.output_activation))
            chromosome.append(random.choice(self.loss_function))
            chromosome.append(random.choice(self.optimiser_function))
        
        return chromosome
    
    
    # Mutate the offspring self.population of self.population_size
    # Mutation occurs by randomly selecting a gene in each chromosome to generate a new random value for
    def mutate_population(self):
        # For each chromosome in self.population
        for chromosome in self.population:
            # Randomly select a gene in the chromosome to mutate
            gene = random.randrange(len(chromosome))     
            
            if self.classifier == 'DecisionTreeClassifier':
                if gene == 0:
                    if chromosome[0] == 'gini':
                        chromosome[0] = 'entropy'
                    else:
                        chromosome[0] = 'gini'
                elif gene == 1:
                    if chromosome[1] == 'best':
                        chromosome[1] = 'random'
                    else:
                        chromosome[1] = 'best'
                elif gene == 2:
                    chromosome[2] = random.randrange(self.min_samples_splits['low'], self.min_samples_splits['high'])
                elif gene == 3:
                    chromosome[3] = random.randrange(self.min_samples_leafs['low'], self.min_samples_leafs['high'])
                elif gene == 4:
                    chromosome[4] = random.uniform(self.min_weight_fraction_leafs['low'], self.min_weight_fraction_leafs['high'])
                elif gene == 5:
                    if chromosome[5] == 'balanced':
                        chromosome[5] = None
                    else:
                        chromosome[5] = 'balanced'
            
            elif self.classifier == 'RandomForestClassifier':
                if gene == 0:
                    chromosome[0] = random.randrange(self.n_estimators['low'], self.n_estimators['high'])
                elif gene == 1:
                    if chromosome[1] == 'gini':
                        chromosome[1] = 'entropy'
                    else:
                        chromosome[1] = 'gini'
                elif gene == 2:
                    chromosome[2] = random.randrange(self.min_samples_splits['low'], self.min_samples_splits['high'])
                elif gene == 3:
                    chromosome[3] = random.randrange(self.min_samples_leafs['low'], self.min_samples_leafs['high'])
                elif gene == 4:
                    chromosome[4] = random.uniform(self.min_weight_fraction_leafs['low'], self.min_weight_fraction_leafs['high'])
                elif gene == 5:
                    if chromosome[5] == 'balanced':
                        chromosome[5] = None
                    else:
                        chromosome[5] = 'balanced'    

            elif self.classifier == 'KNeighborsClassifier':
                if gene == 0:
                    chromosome[0] = random.randrange(self.n_neighbors['low'], self.n_neighbors['high'])
                elif gene == 1:
                    if chromosome[1] == 'uniform':
                        chromosome[1] = 'distance'
                    else:
                        chromosome[1] = 'uniform'
                elif gene == 2:
                    if chromosome[2] == 'ball_tree':
                        chromosome[2] = 'kd_tree'
                    else:
                        chromosome[2] = 'ball_tree'                                        
                elif gene == 3:
                    chromosome[3] = random.randrange(self.leaf_size['low'], self.leaf_size['high'])
                elif gene == 4:
                    chromosome[4] = random.randrange(self.p['low'], self.p['high'])

            elif self.classifier == 'AdaBoostClassifier':
                if gene == 0:
                    chromosome[0] = random.randrange(self.n_estimators['low'], self.n_estimators['high'])
                elif gene == 1:
                    chromosome[1] = random.uniform(self.learning_rate['low'], self.learning_rate['high'])
                elif gene == 2:
                    if chromosome[2] == 'SAMME':
                        chromosome[2] = 'SAMME.R'
                    else:
                        chromosome[2] = 'SAMME'
                elif gene == 3:
                    chromosome[3] = random.randrange(self.random_state['low'], self.random_state['high'])

            elif self.classifier == 'LinearSVC':
                if gene == 0:
                    if chromosome[0] == 'hinge':
                        chromosome[0] = 'squared_hinge'
                    else:
                        chromosome[0] = 'hinge'
                elif gene == 1:
                    chromosome[1] = random.uniform(self.tol['low'], self.tol['high'])
                elif gene == 2: 
                    chromosome[2] = random.randrange(self.c['low'], self.c['high'])

            # Keras Deep Learning FFNN
            elif self.classifier == 'FFNN':
                if gene == 0:
                    chromosome[0] = random.randrange(self.epochs['low'], self.epochs['high'])
                elif gene == 1:
                    chromosome[1] = random.randrange(self.batch_size['low'], self.batch_size['high'])
                elif gene == 2:
                    chromosome[2] = random.randrange(self.layer_units['low'], self.layer_units['high'])
                elif gene == 3:
                    chromosome[3] = random.choice(self.layers_activation)
                elif gene == 4:
                    chromosome[4] = random.choice(self.output_activation)
                elif gene == 5:
                    chromosome[5] = random.choice(self.loss_function)
                elif gene == 6:
                    chromosome[6] = random.choice(self.optimiser_function)
                    
            elif self.classifier == 'FFNN_2':
                if gene == 0:
                    chromosome[0] = random.randrange(self.epochs['low'], self.epochs['high'])
                elif gene == 1:
                    chromosome[1] = random.randrange(self.batch_size['low'], self.batch_size['high'])
                elif gene == 2:
                    chromosome[2] = random.randrange(self.layer_units['low'], self.layer_units['high'])
                elif gene == 3:
                    chromosome[3] = random.choice(self.layers_activation)
                elif gene == 4:
                    chromosome[4] = random.uniform(self.dropout_1['low'], self.dropout_1['high'])
                elif gene == 5:
                    chromosome[5] = random.randrange(self.layer_units_2['low'], self.layer_units_2['high'])
                elif gene == 6:
                    chromosome[6] = random.randrange(self.input_dim_2['low'], self.input_dim_2['high'])
                elif gene == 7:
                    chromosome[7] = random.choice(self.layers_activation_2)
                elif gene == 8:
                    chromosome[8] = random.uniform(self.dropout_2['low'], self.dropout_2['high'])
                elif gene == 9:
                    chromosome[9] = random.choice(self.output_activation)
                elif gene == 10:
                    chromosome[10] = random.choice(self.loss_function)
                elif gene == 11:
                    chromosome[11] = random.choice(self.optimiser_function)




    # Plot each generation's best fitness score
    def plot_fitness_scores(self):
        fig = plt.figure()
        ax = plt.axes()
        
        # Plots the best fitness score from each generation
        plt.plot(self.best_generation_fitness_scores, c='g')
            
        plt.show()            
