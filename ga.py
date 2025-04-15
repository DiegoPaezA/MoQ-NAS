import os
import time
import datetime
import numpy as np
import evaluation
import qnas_config as cfg
from pickle import dump, load, HIGHEST_PROTOCOL
from util import delete_old_dirs, init_log, check_files, download_dataset


class GA(object):
    """
    """
    def __init__(self, eval_func, experiment_path, log_file, log_level, data_file):
        """
        Initialize the GA evolution.

        Args:
            eval_func: Callable to evaluate a candidate architecture.
                Must have signature: eval_func(decoded_individual, generation)
            experiment_path: Path to folder for saving logs/models.
            log_file: Filename for log output.
            log_level: Logging level, e.g., "INFO", "DEBUG".
            data_file: Path to file for saving evolution data (pickle).
        """
        # GA parameters (to be set later via initialize_ga)
        self.population_size = None
        self.num_generations = None
        self.crossover_rate = None
        self.mutation_rate = None
        self.elitism = False

        # GA state variables
        self.population = None
        self.fitnesses = None
        self.best_so_far = -np.inf
        self.best_so_far_id = None
        self.current_gen = 0
        self.total_eval = 0

        # Early stopping parameters
        self.patience = None
        self.early_stopping_counter = 0

        self.eval_func = eval_func
        self.experiment_path = experiment_path
        self.data_file = data_file

        # Create a logger using the provided utility function (or basicConfig)
        self.logger = init_log(log_level, log_file)
    

    def initialize_ga(self, population_size, num_generations, max_num_nodes,fn_list,
                    crossover_rate, mutation_rate, elitism=False, patience=20):
        """
        Initialize GA parameters and create the initial random population.

        Args:
            population_size: (int) number of individuals.
            num_generations: (int) maximum number of generations.
            max_num_nodes: (int) length of each individual chromosome.
            crossover_rate: (float) probability for crossover.
            mutation_rate: (float) probability for mutation (per gene).
            fn_list: List of function names (e.g., layer types) to decode chromosomes.
            elitism: (bool) whether to keep the best individual in the next generation.
            patience: (int) number of generations with little/no improvement to trigger early stopping.
        """
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.patience = patience
        self.fn_list = fn_list  # Store the list of function names

        # Initialize a population of individuals.
        # Here we assume a discrete/categorical representation.
        # You can modify "random_individual" to suit your encoding.
        def random_individual(length):
            # Example: each gene takes a random integer from 0 to 9
            return np.random.randint(0, len(self.fn_list), size=length).tolist()

        pop_list = [random_individual(max_num_nodes) for _ in range(population_size)]
        self.population = np.array(pop_list)
        self.logger.info("Initial population created with size: %d", population_size)
        
    def decode_net(self, chromosome):
        """
        Convert a chromosome (list or numpy array of integers) into the corresponding
        list of function names representing the network layers.

        Args:
            chromosome: list or numpy array, where each element is an integer index.

        Returns:
            List with function names in the order they represent the network.
        """
        # Using len(chromosome) if chromosome is a list; if it's a numpy array, chromosome.shape[0] works as well.
        decoded = [None] * len(chromosome)
        for i, gene in enumerate(chromosome):
            if gene >= 0:
                decoded[i] = self.fn_list[gene]
        return decoded
    
    def decode_pop(self, pop_params, pop_net):
        """ Decode a population of parameters and networks.

        Args:
            pop_params: float numpy array with a classic population of hyperparameters.
            pop_net: int numpy array with a classic population of networks.

        Returns:
            list of decoded params and list of decoded networks.
        """

        num_individuals = pop_net.shape[0]

        #decoded_params = [None] * num_individuals
        decoded_nets = [None] * num_individuals

        for i in range(num_individuals):
            #decoded_params[i] = self.qpop_params.chromosome.decode(pop_params[i])
            decoded_nets[i] = self.decode_net(pop_net[i, :])

        #return decoded_params, decoded_nets
        return decoded_nets
    
    def evaluate_population(self):
        """
        Evaluate the current population using the provided eval_func.
        The eval_func should return a list or array of fitness values for all individuals.
        """

        decoded_net = self.decode_pop(None, self.population)
        # Create the backbone_percentage variable (a NumPy array) with shape (population_size, 1) # test
        backbone_percentage_array = np.array([np.random.uniform(0.0, 1.0) for _ in range(len(decoded_net))]).reshape(-1, 1)

        # Package the variable into a dictionary, then into a list
        decoded_params = [{'backbone_percentage': backbone_percentage_array[i]} for i in range(len(backbone_percentage_array))]
        
        fitnesses = self.eval_func(decoded_params, decoded_net, generation=self.current_gen)
        
        self.fitnesses = np.array(fitnesses)
        self.update_best_id(self.fitnesses)
        self.logger.info("Generation %d evaluated; best fitness so far: %.4f", 
                        self.current_gen, self.best_so_far)
        return self.fitnesses

    def update_best_id(self, fitnesses):
        """
        Update the best individual id based on current fitnesses.
        Args:
            fitnesses: numpy array of current population fitnesses.
        """
        best_idx = np.argmax(fitnesses)
        best_fitness = fitnesses[best_idx]
        if best_fitness > self.best_so_far:
            self.best_so_far = best_fitness
            self.best_so_far_id = (self.current_gen, best_idx)
        self.logger.debug("Updated best individual at generation %d: index %d with fitness %.4f",
                        self.current_gen, best_idx, best_fitness)

    def selection(self):
        """
        Select an individual from the population using tournament selection.
        Tournament size is set to 3 by default.
        """
        tournament_size = 3
        participants = np.random.choice(range(self.population_size), tournament_size, replace=False)
        best = participants[0]
        for idx in participants:
            if self.fitnesses[idx] > self.fitnesses[best]:
                best = idx
        return self.population[best]

    def crossover(self, parent1, parent2):
        """
        Perform uniform crossover between two parents.
        Returns two offspring.
        """
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        for i in range(len(parent1)):
            if np.random.rand() < 0.5:
                offspring1[i], offspring2[i] = parent2[i], parent1[i]
        return offspring1, offspring2

    def mutate(self, individual):
        """
        Mutate an individual by randomly changing genes based on mutation_rate.
        """
        for i in range(len(individual)):
            if np.random.rand() < self.mutation_rate:
                # Mutation: assign a new random gene value in the assumed range (0 to 9)
                individual[i] = np.random.randint(0, 10)
        # You can also implement more complex mutation strategies here.
        # For example, swap two genes, or add noise to a continuous variable.
        # Here we just replace the gene with a new random value.
        # This is a simple mutation strategy.

        return individual

    def generate_offspring(self):
        """
        Create a new population using selection, crossover and mutation.
        Applies crossover with self.crossover_rate and mutation with self.mutation_rate.
        If elitism is enabled, the best individual is preserved.
        """
        new_population = []
        # Optionally preserve best individual (elitism)
        if self.elitism:
            best_individual = self.population[np.argmax(self.fitnesses)]
            new_population.append(best_individual)

        while len(new_population) < self.population_size:
            # Parent selection (tournament selection)
            parent1 = self.selection()
            parent2 = self.selection()

            # Crossover operator
            if np.random.rand() < self.crossover_rate:
                child1, child2 = self.crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Mutation operator
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)

            new_population.extend([child1, child2])
        # Trim in case population exceeded population_size
        self.population = np.array(new_population[:self.population_size])
        self.logger.info("Offspring generated for generation %d", self.current_gen)
    
    def log_data(self):
        """ Log GA evolution statistics such as generation number and fitnesses. """
        self.logger.info("Generation %d: Best Fitness = %.4f; Mean Fitness = %.4f",
                        self.current_gen, self.best_so_far, np.mean(self.fitnesses))

    def save_data(self):
        """
        Save current evolution data to self.data_file (pickle format)
        so that the evolution process may be resumed or analysed.
        """
        data = {
            'current_gen': self.current_gen,
            'total_eval': self.total_eval,
            'best_so_far': self.best_so_far,
            'best_so_far_id': self.best_so_far_id,
            'population': self.population,
            'fitnesses': self.fitnesses
        }
        with open(self.data_file, 'wb') as f:
            dump(data, f, protocol=HIGHEST_PROTOCOL)
        self.logger.info("Evolution data saved at generation %d", self.current_gen)

    def load_data(self, file_path):
        """
        Load evolution data from a pickle file.
        Args:
            file_path: path to the pickle file with saved data.
        """
        with open(file_path, 'rb') as f:
            data = load(f)
        self.current_gen = data['current_gen']
        self.total_eval = data['total_eval']
        self.best_so_far = data['best_so_far']
        self.best_so_far_id = data['best_so_far_id']
        self.population = data['population']
        self.fitnesses = data['fitnesses']
        self.logger.info("Evolution data loaded from %s at generation %d", file_path, self.current_gen)
    
    def check_early_stopping(self):
        """
        Check for early stopping criteria. If no improvement in best fitness is observed
        for self.patience generations, return True.
        """
        if self.current_gen == 0:
            self.last_best = self.best_so_far
            return False

        improvement = (self.best_so_far - self.last_best) / (abs(self.last_best) + 1e-9)
        if improvement < 0.005:  # less than 0.5% improvement
            self.early_stopping_counter += 1
            self.logger.info("No significant improvement (%d generations without improvement)",
                            self.early_stopping_counter)
            if self.early_stopping_counter >= self.patience:
                self.logger.info("Early stopping triggered at generation %d", self.current_gen)
                return True
        else:
            self.early_stopping_counter = 0
        self.last_best = self.best_so_far
        return False

    def go_next_gen(self):
        """
        Perform end-of-generation routines: log data, save data, and update generation counter.
        """
        self.log_data()
        self.save_data()
        delete_old_dirs(self.experiment_path, keep_best=True,
                        best_id=f'{self.best_so_far_id[0]}_{self.best_so_far_id[1]}')
        self.current_gen += 1

    def evolve(self):
        """
        Main evolution loop. Repeats until maximum generations is reached or early stopping is triggered.
        """
        start_time = time.time()
        self.logger.info("Starting evolution for %d generations", self.num_generations)
        
        while self.current_gen < self.num_generations:
            self.logger.info("Generation %d started", self.current_gen)
            # Evaluate the current population
            self.evaluate_population()
            # Generate offspring using selection, crossover, and mutation
            self.generate_offspring()
            # End-of-generation update
            self.go_next_gen()
            # # Optionally check for early stopping
            # if self.check_early_stopping():
            #     break
        
        total_time = time.time() - start_time
        hours, rem = divmod(total_time, 3600)
        minutes, _ = divmod(rem, 60)
        self.logger.info("Evolution completed in %.0f hours %.0f minutes", hours, minutes)
        self.logger.info("Best solution found at generation %d with fitness %.4f",
                        self.best_so_far_id[0] if self.best_so_far_id else -1, self.best_so_far)
        return self.population, self.fitnesses, self.best_so_far, self.best_so_far_id

# Example usage:
if __name__ == "__main__":
    # Define parameters similar to those defined in your .sh script
    args = {
        "experiment_path": "experiment_cifar10_ga/exp1_repeat_1",
        "data_path": "cifar10_data",
        "dataset": "cifar10",
        "config_file": "config_files_cifar/config0.txt",
        "continue_path": "",
        "log_level": "DEBUG",  # Using DEBUG for detailed output during development
        "optimizer": "AdamW",
        "fitness_metric": "best_accuracy",
        "data_augmentation": False,  # Set as needed for your experiment
        "early_stopping": True,
        "en_pop_crossover": True,
        "save_checkpoints_epochs": 5,
        "limit_data_value": 10000,
        "backbone_name": "resnet18",
        "network_config": "default",
    }

    logger = init_log(args['log_level'], name=__name__)

    if not os.path.exists(args['experiment_path']):
        logger.info(f"Creating {args['experiment_path']} ...")
        os.makedirs(args['experiment_path'])

    # Evolution or continue previous evolution
    if not args['continue_path']:
        phase = 'evolution'
    else:
        phase = 'continue_evolution'
        logger.info(f"Continue evolution from: {args['continue_path']}. Checking files ...")
        check_files(args['continue_path'])

    logger.info(f"Getting parameters from {args['config_file']} ...")
    config = cfg.ConfigParameters(args, phase=phase)
    config.get_parameters()
    logger.info(f"Saving parameters for {config.phase} phase ...")
    config.save_params_logfile()
    
    if config.train_spec['mixed_precision']:
        logger.info(f"Using mixed precision training ...")
        
    # Download dataset
    dataset_status = download_dataset(params=config.train_spec)
    status_message = "Dataset is already downloaded." if dataset_status else "Dataset downloaded successfully."
    logger.info(status_message)
    
    eval_pop = evaluation.EvalPopulation(params=config.train_spec,
                                                fn_dict=config.fn_dict,
                                                log_level=config.train_spec['log_level'])
    # Initialize GA instance
    ga = GA(eval_pop, config.train_spec['experiment_path'], 
            config.files_spec['log_file'], 
            log_level=config.train_spec['log_level'], 
            data_file=config.files_spec['data_file'])
    # Set GA parameters: population_size, num_generations, max_num_nodes, crossover_rate, mutation_rate, etc.
    ga.initialize_ga(population_size=5, num_generations=3, max_num_nodes=10,
                    crossover_rate=0.7, mutation_rate=0.1, elitism=True, patience=10, fn_list=config.QNAS_spec['fn_list'])
    
    # Run the evolution
    population, fitnesses, best_fitness, best_id = ga.evolve()
    print("Best fitness:", best_fitness)
    print("Best individual (at generation, index):", best_id)
