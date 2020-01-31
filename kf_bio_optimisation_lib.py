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
        else:
            raise Exception("Classifier not supported: ", self.classifier)
                
        
        if self.initial_pop_gen_method not in ['heuristic', 'random']:
            raise Exception("Initial Population Generation Method not supported: ", self.initial_pop_gen_method)
        #print("Inital population geneneration method = ", self.initial_pop_gen_method)

        
        # Generate inital chromosome population, with size population_size
        self.population = self.generate_inital_population()
        
        
        # Load, process and split dataset
        self.dataset = kf.load_dataset(dataset_path)
        self.X, self.y = kf.split_dataset(self.dataset, extended=self.extended_dataset)
        
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
        #print("Generating Population of size: ", self.population_size)
        
        # Inital chromosome population
        population = []
        start = 0
        
        # First chromosome with default parameters, if initial_pop_gen_method=='heuristic'
        if self.initial_pop_gen_method == 'heuristic':
            start = 1
            chromosome = []
            
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
                chromosome.append(1)
                chromosome.append(None)
            elif self.classifier == 'KNeighborsClassifier':
                chromosome.append(5)
                chromosome.append('uniform')
                chromosome.append('kd_tree')
                chromosome.append(30)
                chromosome.append(2)
        
            population.append(chromosome)
        
        
        # Fill each chromosome with randomly selected genes from the classifier's gene pools 
        # and add it to the population
        for i in range(start, self.population_size):
            chromosome = self.generate_random_chromosome()
            
            population.append(chromosome)
        
        #print("Generated Inital Population:")
        #print(population)
        
        return population

    
    # Run the Genetic Algorithm for N generations
    def run(self):        
        for generation in range(0, self.generations):
            #print("\nGENERATION = ", generation)

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
            
            #print("\nOFFSPRING POPULATION:")
            #for chromosome in self.population:
            #    print(chromosome)
            
            # Mutate the population
            self.mutate_population()
            
            #print("\nNew Population:")
            #for chromosome in self.population:
            #     print(chromosome)
        
        
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
                clf = RandomForestClassifier(n_jobs=8,
                                             n_estimators=chromosome[0],
                                             criterion=chromosome[1],
                                             min_samples_split=chromosome[2],
                                             min_samples_leaf=chromosome[3],
                                             min_weight_fraction_leaf=chromosome[4],
                                             class_weight=chromosome[5])
            elif self.classifier == 'KNeighborsClassifier':
                clf = KNeighborsClassifier(n_jobs=8,
                                           n_neighbors=chromosome[0],
                                           weights=chromosome[1],
                                           algorithm=chromosome[2],
                                           leaf_size=chromosome[3],
                                           p=chromosome[4])

                
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
                
                results = cross_validate(clf, self.X, self.y, cv=10, scoring=scoring, n_jobs=-1, verbose=0)

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
            
        #print("score, parent1 = ", parent1)
        #print("score, parent2 = ", parent2)
        
        self.best_generation_fitness_scores.append(parent1[0])
        self.best_generation_chromosomes.append(parent1[1])
        
        # Remove fitness scores from chromosome
        parent1 = parent1[1]
        parent2 = parent2[1]

        return parent1, parent2
        
        
    # Performs one-point-crossover of the parents, returning 
    def crossover(self, parent1, parent2):       
        if self.classifier == 'DecisionTreeClassifier' or self.classifier == 'RandomForestClassifier':
            offspring1 = [parent1[0], parent1[1], parent1[2], parent2[3], parent2[4], parent2[5]]
            offspring2 = [parent2[0], parent2[1], parent2[2], parent1[3], parent1[4], parent1[5]]
        if self.classifier == 'KNeighborsClassifier':
            offspring1 = [parent1[0], parent1[1], parent1[2], parent2[3], parent2[4]]
            offspring2 = [parent2[0], parent2[1], parent2[2], parent1[3], parent1[4]]
        
        #print("Offspring1 = ", offspring1)
        #print("Offspring2 = ", offspring2)
        
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
            
        #print("RANDOMLY GENERATED CHROMOSOME:")
        #print(chromosome)
        
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

    # Plot each generation's best fitness score
    def plot_fitness_scores(self):
        fig = plt.figure()
        ax = plt.axes()
        
        # Plots the best fitness score from each generation
        plt.plot(self.best_generation_fitness_scores, c='g')
            
        plt.show()
            