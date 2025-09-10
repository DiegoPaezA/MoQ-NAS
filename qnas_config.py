""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Q-NAS configuration.
"""

import inspect
import os
from collections import OrderedDict

import numpy as np
import yaml
import re

from chromosome import QChromosomeNetwork, QChromosomeParams
from cnn import model
from util import load_yaml, load_pkl, natural_key


class ConfigParameters(object):
    """Handles the loading, validation, and organization of all configuration parameters."""

    def __init__(self, args: dict, phase: str):
        """Initializes the ConfigParameters object.

        Args:
            args (dict): A dictionary of command-line arguments.
            phase (str): The current operational phase, which must be one of
                        'evolution', 'continue_evolution', or 'retrain'.
        """
        self.phase = phase
        self.args = args
        self.QNAS_spec = {}
        self.train_spec = {}
        self.files_spec = {}
        self.fn_dict = {}
        self.previous_params_file = None
        self.data_info = None
        self.evolved_params = None

    def _check_vars(self, config_file: dict):
        """Validates the structure and types of parameters in the configuration file.

        This method ensures that all required keys are present in the config file
        and that their corresponding values have the correct data types. It also
        performs value-range checks for specific parameters like learning rates
        and function probabilities.

        Args:
            config_file (dict): The loaded configuration from the YAML file.

        Raises:
            KeyError: If a required variable is missing.
            TypeError: If a variable has an incorrect type.
            ValueError: If a parameter's value is out of its allowed bounds.
        """

        def check_params_ranges():
            """Checks if hyperparameter search ranges are within safe, predefined limits."""
            ranges = config_file['QNAS']['params_ranges']
            allowed = {'decay': (1e-6, 1.0), 'learning_rate': (1e-6, 1.0),
                        'momentum': (0.0, 1.0), 'weight_decay': (1e-10, 1e-1),
                        'backbone_percentage': (0.1, 1.0)}

            for key, value in ranges.items():
                limit = allowed.get(key)
                if not limit: continue
                
                low, high = (value[0], value[1]) if isinstance(value, list) else (value, value)
                if not (limit[0] <= low and high <= limit[1]):
                    raise ValueError(f'{key} value out of bound {limit}!')

        def check_fn_dict():
            """Validates the function dictionary, checking for valid function names and probabilities."""
            available_fn = [c[0] for c in inspect.getmembers(model, inspect.isclass)]
            fn_dict = config_file['QNAS']['function_dict']
            probs = []

            for name, definition in fn_dict.items():
                if definition['function'] not in available_fn:
                    raise ValueError(f"{definition['function']} is not a valid function!")
                for param in definition['params'].values():
                    if not isinstance(param, int) or param < 0:
                        raise ValueError(f"{name} has an invalid parameter: {definition['params']}!")
                
                prob_val = definition['prob']
                probs.append(eval(prob_val) if isinstance(prob_val, str) else prob_val)

            if any(p is not None for p in probs):
                prob_sum = np.sum([p for p in probs if p is not None])
                if not np.isclose(prob_sum, 1.0):
                    raise ValueError(f"Function probabilities should sum to 1.0, but sum to {prob_sum}.")

        vars_dict = {
            'QNAS': [('crossover_rate', float), ('max_generations', int), ('max_num_nodes', int),
                    ('num_quantum_ind', int), ('penalize_number', int), ('repetition', int),
                    ('replace_method', str), ('update_quantum_rate', float), ('update_quantum_gen', int),
                    ('save_data_freq', int), ('params_ranges', dict), ('patience', int),
                    ('crossover_frequency', int), ('pop_crossover_rate', float), ('pop_crossover_method', str),
                    ('elite_mode', str), ('k_elites', int), ('pool_factor', int),
                    ('ema_beta', float), ('rank_weighting', bool),
                    ('initial_prob_distribution', str), ('function_dict', dict),
                    ('terminal_op_name', str), ('pool_op_name', str), ('min_active_len', int),
                    ('truncate_after_noop', bool), ('avoid_consecutive_pool', bool), ('enforce_noop_in_update', bool),
                    ('noop_max_prob', float), ('noop_ramp_cap', bool)],
            'train': [('batch_size', int), ('eval_batch_size', int), ('max_epochs', int),
                    ('epochs_to_eval', int), ('optimizer', str), ('device', str),
                    ('dataset', str), ('mixed_precision', bool), ('fitness_metric', str),
                    ('mo_metric_base', str), ('data_augmentation', bool), ('subtract_mean', bool),
                    ('limit_data', bool), ('limit_data_value', int), ('backbone_name', str),
                    ('network_config', str), ('save_checkpoints_epochs', int),
                    ('save_summary_epochs', float), ('multi_objective', bool),
                    ('objectives', list), ('threads', int)]
        }

        for config, items in vars_dict.items():
            for var_name, var_type in items:
                var = config_file[config].get(var_name)
                if var is None:
                    raise KeyError(f"Variable \"{config}:{var_name}\" not found in config file.")
                if not isinstance(var, var_type):
                    raise TypeError(f"Variable {var_name} should be {var_type} but is {type(var)}")
        
        check_params_ranges()
        check_fn_dict()

        if config_file['train']['epochs_to_eval'] >= config_file['train']['max_epochs']:
            raise ValueError('Invalid epochs_to_eval! It should be < max_epochs.')

    def _get_evolution_params(self):
        """Loads and organizes parameters for a new evolution run."""
        config_file = load_yaml(self.args['config_file'])
        self._check_vars(config_file)

        self.train_spec = dict(config_file['train'])
        self.QNAS_spec = dict(config_file['QNAS'])
        self.files_spec['config_file'] = self.args['config_file']

        ranges = self._get_ranges(config_file)
        self.QNAS_spec['params_ranges'] = OrderedDict(sorted(ranges.items()))
        self.QNAS_spec['early_stopping'] = self.args.get('early_stopping')
        self.QNAS_spec['en_pop_crossover'] = self.args.get('en_pop_crossover')
        self.QNAS_spec['elite_mode'] = self.args.get('elite_mode', 'global_k')
        self.QNAS_spec['truncate_after_noop'] = self.args.get('truncate_after_noop', True)
        self.QNAS_spec['avoid_consecutive_pool'] = self.args.get('avoid_consecutive_pool', True)
        self.QNAS_spec['enforce_noop_in_update'] = self.args.get('enforce_noop_in_update', True)
        if self.train_spec['multi_objective']:
            self.QNAS_spec['ref_dir_method'] = self.args.get('ref_dir_method', 'das-dennis')
        self._get_fn_spec()

        train_override_keys = [
            'fitness_metric', 'optimizer', 'data_augmentation', 'network_config',
            'backbone_name', 'save_checkpoints_epochs', 'dataset', 'data_path',
            'limit_data_value', 'multi_objective', 'objectives'
        ]
        for key in train_override_keys:
            val = self.args.get(key)
            if val is not None:
                self.train_spec[key] = val

        self.train_spec['experiment_path'] = self.args['experiment_path']

    def _get_fn_spec(self):
        """Parses the function dictionary from the config to prepare it for QNAS.

        This method extracts the list of function names, their initial probabilities
        (based on the `initial_prob_distribution` setting), and identifies which
        functions are considered "reducing" based on their stride.
        """
        self.QNAS_spec['fn_list'] = sorted(self.QNAS_spec['function_dict'].keys(), key=natural_key)
        self.fn_dict = self.QNAS_spec.pop('function_dict')
        self.QNAS_spec['initial_probs'] = []
        self.QNAS_spec['reducing_fns_list'] = []

        prob_dist_method = self.QNAS_spec.get('initial_prob_distribution', 'from_config')
        self.QNAS_spec.pop('initial_prob_distribution', None)

        if prob_dist_method == 'from_config':
            print("INFO: Initializing probabilities from the configuration file.")
            for fn in self.QNAS_spec['fn_list']:
                prob = self.fn_dict[fn]['prob']
                self.QNAS_spec['initial_probs'].append(eval(prob) if isinstance(prob, str) else prob)
        elif prob_dist_method == 'uniform':
            print("INFO: Using a uniform initial probability distribution.")
        else:
            raise ValueError(f"Unknown initial_prob_distribution: '{prob_dist_method}'.")

        for fn in self.QNAS_spec['fn_list']:
            if self.fn_dict[fn]['params'].get('strides', 1) > 1:
                self.QNAS_spec['reducing_fns_list'].append(fn)
            del self.fn_dict[fn]['prob']

    def _get_ranges(self, config_file: dict) -> dict:
        """Extracts the hyperparameter search ranges from the config file.

        It filters parameters that are defined as a range (a list) for evolution,
        while setting fixed-value parameters directly in the training specification.

        Args:
            config_file (dict): The loaded configuration from the YAML file.

        Returns:
            dict: A dictionary containing only the parameters to be evolved.
        """
        all_ranges = config_file['QNAS']['params_ranges']
        optimizer = self.train_spec['optimizer']
        
        evolved_ranges = {k: v for k, v in all_ranges.items() if isinstance(v, list)}
        
        if optimizer == 'Momentum':
            evolved_ranges.pop('decay', None)

        for key, value in all_ranges.items():
            if not isinstance(value, list):
                self.train_spec[key] = value

        return evolved_ranges

    def _get_continue_params(self):
        """Loads parameters to continue a previous evolution run."""
        self.files_spec['continue_path'] = self.args['continue_path']
        self.files_spec['previous_QNAS_params'] = os.path.join(
            self.files_spec['continue_path'], 'log_params_evolution.txt')
        self.files_spec['previous_data_file'] = os.path.join(self.args['continue_path'], 'data_QNAS.pkl')

        self.load_old_params()
        self.QNAS_spec['max_generations'] = load_yaml(self.args['config_file'])['QNAS']['max_generations']
        self.train_spec['experiment_path'] = self.args['experiment_path']

    def _get_retrain_params(self):
        """Loads parameters to retrain a specific evolved architecture."""
        self.files_spec['previous_QNAS_params'] = os.path.join(self.args['experiment_path'], 'log_params_evolution.txt')
        self.load_old_params()

        for key, val in self.args.items():
            if key in self.train_spec:
                self.train_spec[key] = val

        self.train_spec['experiment_path'] = os.path.join(self.train_spec['experiment_path'], self.args['retrain_folder'])
        del self.args['retrain_folder']

    def _get_common_params(self):
        """Sets up parameters that are common across all operational phases."""
        self.train_spec['data_path'] = self.args['data_path']
        self.train_spec['phase'] = self.phase
        self.train_spec['log_level'] = self.args['log_level']
        self.files_spec['log_file'] = os.path.join(self.args['experiment_path'], 'log_QNAS.txt')
        self.files_spec['data_file'] = os.path.join(self.args['experiment_path'], 'data_QNAS.pkl')
        
    def get_parameters(self):
        """Main entry point to get all configuration parameters for the specified phase.

        This method dispatches to the appropriate helper function based on the
        `self.phase` and then finalizes the configuration.
        """
        if self.phase == 'evolution':
            self._get_evolution_params()
        elif self.phase == 'continue_evolution':
            self._get_continue_params()
        else:
            self._get_retrain_params()
        self._get_common_params()

    def load_old_params(self):
        """Loads and restores the state from a previous experiment's parameter log file."""
        previous_params_file = load_yaml(self.files_spec['previous_QNAS_params'])
        self.train_spec = dict(previous_params_file['train'])
        self.QNAS_spec = dict(previous_params_file['QNAS'])
        self.QNAS_spec['params_ranges'] = eval(self.QNAS_spec['params_ranges'])
        self.fn_dict = previous_params_file['fn_dict']
    
    def load_evolved_data(self, experiment_path: str):
        """Loads the best evolved architecture from a completed experiment path.

        It locates the best individual by finding the 'best_so_far' symbolic link
        or by parsing folder names, then reads the 'training_params.txt' file
        to retrieve the architecture and its metadata.

        Args:
            experiment_path (str): The path to the completed experiment directory.
        """
        best_so_far_link = os.path.join(experiment_path, 'best_so_far')
        
        if os.path.islink(best_so_far_link):
            best_result_folder = os.path.realpath(best_so_far_link)
        else:
            exp_folders = [f.path for f in os.scandir(experiment_path) if f.is_dir() and f.name[0].isdigit()]
            best_result_folder = sorted(exp_folders, key=natural_key)[-1] if exp_folders else None

        if not best_result_folder:
            raise FileNotFoundError("Could not find a valid result folder in the experiment path.")
            
        params_path = os.path.join(best_result_folder, 'training_params.txt')
        with open(params_path, 'r') as file:
            info = yaml.safe_load(file)
        
        gen, ind = info.get('generation', 0), info.get('individual', 0)
        
        if gen == 0 and ind == 0:  # Fallback for older formats
            matches = re.search(r'(\d+)_(\d+)$', os.path.basename(best_result_folder))
            if matches:
                gen, ind = int(matches.group(1)), int(matches.group(2))

        self.evolved_params = {
            'params': None,
            'net': info.get('net_list', []),
            'generation': gen,
            'individual': ind,
            'backbone_name': info.get('backbone_name'),
            'backbone_percentage': info.get('backbone_percentage', 0)
        }

    def override_train_params(self, new_params_dict: dict):
        """Overrides training parameters with a new set of values.

        Args:
            new_params_dict (dict): A dictionary of parameters to update in `self.train_spec`.
        """
        self.train_spec.update(new_params_dict)

    def params_to_logfile(self, params: dict, text_file, nested_level=0):
        """Recursively writes a dictionary of parameters to a text file with indentation.

        Args:
            params (dict): The dictionary of parameters to write.
            text_file (file object): The file to write to.
            nested_level (int): The current indentation level for pretty-printing.
        """
        spacing = '    '
        for key, value in OrderedDict(sorted(params.items())).items():
            if isinstance(value, dict) and nested_level < 2:
                print(f'{nested_level * spacing}{key}:', file=text_file)
                self.params_to_logfile(value, text_file, nested_level + 1)
            else:
                if isinstance(value, float) and value < 1e-3:
                    formatted_value = f'{value:.2E}'
                elif isinstance(value, float):
                    formatted_value = f'{value:.4f}'
                else:
                    formatted_value = value
                print(f'{nested_level * spacing}{key}: {formatted_value}', file=text_file)
            if nested_level == 0:
                print('', file=text_file)

    def save_params_logfile(self):
        """Saves the final, organized parameters to a log file for reproducibility."""
        if self.phase == 'retrain':
            params_dict = {'evolved_params': self.evolved_params,
                            'train': self.train_spec, 'files': self.files_spec}
        else:
            params_dict = {'QNAS': self.QNAS_spec, 'train': self.train_spec,
                            'files': self.files_spec, 'fn_dict': self.fn_dict}

        params_file_path = os.path.join(self.train_spec['experiment_path'], f'log_params_{self.phase}.txt')
        with open(params_file_path, mode='w') as text_file:
            self.params_to_logfile(params_dict, text_file)