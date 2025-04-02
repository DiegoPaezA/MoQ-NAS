""" Copyright (c) 2025, Diego Páez
* Licensed under the MIT license

- Master module for training and evaluating CNN models.

"""
import os

import torch
import torch.nn as nn

from typing import Dict, List, Union, Any
from cnn import input, model, model_resnet, trainer
from util import init_log, load_yaml

# Initialize a logger (assumed to be defined in init_log)
current_directory = os.path.dirname(os.path.dirname(__file__))
log_directory = os.path.join(current_directory, 'logs')
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

log_file = os.path.join(log_directory, 'master.log')
LOGGER = init_log("INFO", name=__name__, file_path=log_file)


def create_optimizer(net, params):
    """
    Select and create an optimizer based on the configuration options in params.

    Args:
        net (nn.Module): The neural network model.
        params (Dict[str, Any]): Configuration dictionary containing keys such as 'optimizer' 
            and 'learning_rate'.

    Returns:
        torch.optim.Optimizer: An instance of the selected optimizer.
    """
    if params['optimizer'] == 'RMSProp':
        optimizer = torch.optim.RMSprop(net.parameters())
    elif params['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(net.parameters())
    elif params['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(net.parameters())
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=params['learning_rate'])
    return optimizer

def ensure_model_path(params):
    """
    Ensure that a model path exists based on the 'experiment_path' in params,
    and update params with the model path. This is used in the retrain and resnet training phases.

    Args:
        params (Dict[str, Any]): Configuration dictionary containing 'experiment_path'.

    Returns:
        Dict[str, Any]: Updated configuration dictionary with 'model_path' set.
    """
    model_path = os.path.join(params['experiment_path'])
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    params['model_path'] = model_path
    return params

def setup_additional_params(params, id_num=None):
    """
    Set additional parameters such as model path, generation, and individual if an identifier is provided.
    This is useful for the evolution phase where each model has a unique ID.

    Args:
        params (Dict[str, Any]): Configuration dictionary.
        id_num (str, optional): A string identifier in the format "generation_individual".

    Returns:
        Dict[str, Any]: Updated configuration dictionary with additional keys (e.g. 'model_path',
                        'generation', 'individual').
    """
    if id_num is not None:
        model_path = os.path.join(params['experiment_path'], id_num)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        params['model_path'] = model_path
        params['generation'] = id_num.split('_')[0]
        params['individual'] = id_num.split('_')[1]
    return params

def setup_dataset_info(params):
    """
    Update the configuration parameters with dataset-specific information.
    If the dataset is available in input.available_datasets, use that information;
    otherwise, load the dataset info from a YAML file.

    Args:
        params (Dict[str, Any]): Configuration dictionary containing keys 'dataset', 
            'data_path', and 'batch_size'.

    Returns:
        Dict[str, Any]: Updated configuration dictionary with 'num_classes', 'task',
                        and 'input_shape' set.
    """
    if params['dataset'].lower() in input.available_datasets:
        dataset_info = input.available_datasets[params['dataset'].lower()]
    else:
        dataset_info = load_yaml(os.path.join(params['data_path'], 'data_info.txt'))
    
    params['num_classes'] = dataset_info['num_classes']
    params['task'] = dataset_info['task']
    params['input_shape'] = [params['batch_size']] + dataset_info['shape']
    return params

def create_model_and_trainer(params, train_loader, val_loader, test_loader):
    """
    Create the model and corresponding trainer instance based on the training phase.
    For the 'resnet' phase, a ResNet model and ResNetTrainer are used.
    For 'evolution' or 'retrain' phases, a generic NetworkGraph model and BaseTrainer are used.

    Args:
        params (Dict[str, Any]): Configuration dictionary containing keys such as 'phase', 
            'num_classes', 'network_config', 'fn_dict', and 'net_list'.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        test_loader: Test DataLoader (can be None for evolution phase).

    Returns:
        Union[BaseTrainer, ResNetTrainer]: The trainer instance corresponding to the selected phase.
    """

    phase = params.get('phase')
    if phase == 'resnet':
        # Use ResNetTrainer for resnet training phase
        n_channels = params.get('input_channels', 3)
        num_classes = params.get('num_classes', 10)
        model_flag = params.get('model_flag', 'resnet18')
        if model_flag == 'resnet18':
            net = model_resnet.ResNet18(in_channels=n_channels, num_classes=num_classes)
        elif model_flag == 'resnet50':
            net = model_resnet.ResNet50(in_channels=n_channels, num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported model_flag: {model_flag}")
        optimizer = create_optimizer(net, params)
        if params['task'] == 'multi-label, binary-class':
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        trainer_instance = trainer.ResNetTrainer(net, criterion, optimizer,
                                        train_loader, val_loader, test_loader, params)
    else:
        # For evolution or retrain phases, use the generic BaseTrainer
        net = model.NetworkGraph(num_classes=params['num_classes'],
                                network_config=params['network_config'],
                                network_gap=params['network_gap'])
        # Create functions using fn_dict and net_list from params
        filtered_dict = {key: val for key, val in params['fn_dict'].items() if key in params['net_list']}
        has_cbam_key = any(key.startswith('cbam') for key in filtered_dict)
        net.create_functions(fn_dict=filtered_dict, net_list=params['net_list'], cbam=has_cbam_key)
        # Run a dummy input through the network to initialize any fully connected layers
        dummy_input = torch.randn(params['input_shape'])
        with torch.no_grad():
            _ = net(dummy_input)
        optimizer = create_optimizer(net, params)
        criterion = nn.CrossEntropyLoss()
        trainer_instance = trainer.BaseTrainer(net, criterion, optimizer,
                                    train_loader, val_loader, test_loader, params)
    return trainer_instance

def run_training_phase(params: Dict[str, Any],
                        fn_dict: Dict[str, Any] = None,
                        net_list: List[str] = None,
                        id_num: str = None, debug: bool = False,
                        train_loader=None, val_loader=None, test_loader=None) -> Dict[str, Any]:
    """
    Generic function to update parameters, create the trainer, and run training.
    It updates fn_dict, net_list, and additional parameters if provided, ensures that
    the dataset info and (if necessary) the model path are set, and then creates a trainer
    instance based on the phase ('evolution', 'retrain', or 'resnet').

    Args:
        params (Dict[str, Any]): Configuration dictionary.
        fn_dict (Dict[str, Any], optional): Dictionary of layer definitions.
        net_list (List[str], optional): List of layer names to be used in the network.
        id_num (str, optional): Identifier for the model (used in the evolution phase).
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        test_loader: Test DataLoader (can be None for certain phases).

    Returns:
        Dict[str, Any]: Dictionary containing the training results.
    """
    if fn_dict is not None:
        params['fn_dict'] = fn_dict
    if net_list is not None:
        params['net_list'] = net_list
    if id_num is not None:
        params = setup_additional_params(params, id_num=id_num)
        
    # For retrain and resnet, ensure model path is created
    if params['phase'] in ['retrain', 'resnet']:
        params = ensure_model_path(params)
    params = setup_dataset_info(params)

    
    trainer = create_model_and_trainer(params, train_loader, val_loader, test_loader)

    results_dict = trainer.train(debug=debug)
        
    return results_dict

def fitness(id_num: str, params: Dict[str, Any], 
            fn_dict: Dict[str, Any], net_list: List[str],
            decoded_params: Dict[str, Any],
            train_loader: torch.utils.data.DataLoader, 
            val_loader: torch.utils.data.DataLoader,
            return_val, debug: bool = False) -> Dict[str, Any]:
    """
    Train and evaluate a model using evolved networks in the evolution phase.
    Updates the mutable container return_val with key performance metrics based on the fitness_metric in params.

    Args:
        id_num (str): Identifier for the model in the format "generation_individual".
        params (Dict[str, Any]): Configuration dictionary with evolved networks.
        fn_dict (Dict[str, Any]): Dictionary of layer definitions.
        net_list (List[str]): List of layer names to be used in the network.
        train_loader (torch.utils.data.DataLoader): Training DataLoader.
        val_loader (torch.utils.data.DataLoader): Validation DataLoader.
        return_val: A mutable container (e.g., list) to store key metrics.
        debug (bool, optional): If True, returns the raw results dictionary for debugging.

    Returns:
        Dict[str, Any]: Dictionary containing the training results.

    Raises:
        Exception: Propagates any exception encountered during training after setting return_val to zeros.
    """
    print(decoded_params)
    try:
        results_dict = run_training_phase(params, fn_dict, net_list, id_num, debug, train_loader, val_loader, None)
        if debug:
            return results_dict
        else:
            if params['fitness_metric'] == 'best_accuracy':
                return_val[:] = [results_dict['best_accuracy'],
                                results_dict['total_trainable_params'],
                                results_dict['cuda_inference_time']]
            elif params['fitness_metric'] == 'best_loss':
                return_val[:] = [results_dict['fitness_val_loss'],
                                results_dict['total_trainable_params'],
                                results_dict['cuda_inference_time']]
            elif params['fitness_metric'] == 'scalar_multi_objective':
                return_val[:] = [results_dict['scalar_multi_objective'],
                                results_dict['total_trainable_params'],
                                results_dict['cuda_inference_time']]
            else:
                raise ValueError(f"Invalid fitness metric: {params['fitness_metric']}")
            LOGGER.info(f"Training of model {id_num} finished, best {params['fitness_metric']}: {round(return_val[0], 2)}")
            return results_dict
    except Exception as e:
        return_val[:] = [0.0, 0.0, 0.0]
        raise e

def retrain(params: Dict[str, Any],
            fn_dict: Dict[str, Any],
            net_list: List[str],
            train_loader: torch.utils.data.DataLoader, 
            val_loader: torch.utils.data.DataLoader,
            test_loader: torch.utils.data.DataLoader) -> Dict[str, Any]:
    """
    Retrain a model using the best architecture obtained during evolution.
    This method assumes that the evolved parameters (or a best model checkpoint)
    are available in the configuration.

    Args:
        params (Dict[str, Any]): Configuration dictionary.
        fn_dict (Dict[str, Any]): Dictionary of layer definitions.
        net_list (List[str]): List of layer names to be used in the network.
        train_loader (torch.utils.data.DataLoader): Training DataLoader.
        val_loader (torch.utils.data.DataLoader): Validation DataLoader.
        test_loader (torch.utils.data.DataLoader): Test DataLoader.

    Returns:
        Dict[str, Any]: Dictionary containing the retraining results.

    Raises:
        Exception: Propagates any exception encountered during training.
    """
    try:
        results_dict = run_training_phase(params, fn_dict, net_list, None, train_loader, val_loader, test_loader)
        LOGGER.info(f"Retraining finished, best {params['fitness_metric']}: {round(results_dict['best_accuracy'], 2)}")
        return results_dict
    except RuntimeError as e:
        if "out of memory" in str(e):
            LOGGER.error(f"Out of memory error: {e}")
            results_dict = None
        else:
            LOGGER.error(f"Runtime error during training: {e}")
            raise
    except Exception as e:
        LOGGER.error(f"An unexpected error occurred during training: {e}")
        raise e

def resnet_train(params: Dict[str, Any],
                train_loader: torch.utils.data.DataLoader, 
                val_loader: torch.utils.data.DataLoader,
                test_loader: torch.utils.data.DataLoader) -> Dict[str, Any]:
    """
    Train a ResNet model using ResNetTrainer.
    This method is tailored for the 'resnet' phase.

    Args:
        params (Dict[str, Any]): Configuration dictionary.
        train_loader (torch.utils.data.DataLoader): Training DataLoader.
        val_loader (torch.utils.data.DataLoader): Validation DataLoader.
        test_loader (torch.utils.data.DataLoader): Test DataLoader.

    Returns:
        Dict[str, Any]: Dictionary containing the training results.

    Raises:
        Exception: Propagates any exception encountered during training.
    """
    try:
        results_dict = run_training_phase(params, None, None, None, train_loader, val_loader, test_loader)
        LOGGER.info(f"ResNet training finished, best {params['fitness_metric']}: {round(results_dict['best_accuracy'], 2)}")
        return results_dict
    except RuntimeError as e:
        if "out of memory" in str(e):
            LOGGER.error(f"Out of memory error: {e}")
            results_dict = None
            raise
        else:
            LOGGER.error(f"Runtime error during training: {e}")
            raise
    except Exception as e:
        LOGGER.error(f"An unexpected error occurred during training: {e}")
        raise e