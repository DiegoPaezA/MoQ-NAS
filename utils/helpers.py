import logging
import yaml
import pickle as pkl
import os
import re
import json
import tempfile
import time
import matplotlib.pyplot as plt
from shutil import rmtree
import numpy as np
import seaborn as sns
import pandas as pd
import torchvision.datasets
from torchvision.transforms import ToTensor
import medmnist
from medmnist import INFO
from typing import Dict, List, Optional

import shutil


import GPUtil
from pickle import dump, load, HIGHEST_PROTOCOL

import plotly.express as px
from pymoo.indicators.hv import Hypervolume

from collections import defaultdict
import statistics

def natural_key(string):
    """ Key to use with sort() in order to sort string lists in natural order.
        Example: [1_1, 1_2, 1_5, 1_10, 1_13].
    """

    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string)]

def load_yaml(file_path):
    """ Wrapper to load a yaml file.

    Args:
        file_path: (str) path to the file to load.

    Returns:
        dict with loaded parameters.
    """

    if not os.path.exists(file_path):
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}

def _deep_merge(dst: dict, src: dict) -> dict:
    """Deep-merge src into dst (in place)."""
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst

def _atomic_write_yaml(file_path: str, data: dict):
    """Write YAML atomically (temp file + replace)."""
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".params.", suffix=".tmp", dir=os.path.dirname(file_path) or ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                data,
                f,
                sort_keys=False,         # keep human-friendly order if present
                default_flow_style=False,
                allow_unicode=True
            )
        os.replace(tmp_path, file_path)  # atomic on POSIX
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

def update_yaml_file(file_path: str, patch: dict):
    """
    Read existing YAML (or {}), deep-merge `patch`, then atomically write it back.
    """
    current = load_yaml(file_path)
    if not isinstance(current, dict):
        # If the root isn't a mapping, upgrade to a mapping
        current = {"_value": current}
    merged = _deep_merge(current, patch or {})
    _atomic_write_yaml(file_path, merged)

def load_pkl(file_path):
    """ Load a pickle file.

    Args:
        file_path: (str) path to the file to load.

    Returns:
        loaded data.
    """

    with open(file_path, 'rb') as f:
        file = pkl.load(f)

    return file

def create_info_file(out_path, info_dict, file_name='data_info.txt'):
    """ Saves info in *info_dict* in a txt file.

    Args:
        out_path: (str) path to the directory where to save info file.
        info_dict: dict with all relevant info the user wants to save in the info file.
    """
    
    with open(os.path.join(out_path, file_name), 'w') as f:
        yaml.dump(info_dict, f)

def save_results_file(out_path, results_dict, file_name='retrain_results.txt'):
    """ Saves results in *results_dict* in a txt file.

    Args:
        out_path: (str) path to the directory where to save results file.
        results_dict: dict with all relevant results the user wants to save in the results file.
    """

    with open(os.path.join(out_path, file_name), 'w') as f:
        json.dump(results_dict, f, indent=4)

def check_file_exists(file_path):
    """ Check if a file exists.
    
    Args:
        file_path: (str) path to the file to check.
        
    Returns:
        True if the file exists, False otherwise.
    """
    if os.path.exists(file_path):
        return True
    else:
        return False

def load_retrain_results(experiment_path, retrain_file_name):
    file_path = os.path.join(experiment_path, retrain_file_name)
    with open(file_path, 'r') as f:
        retrain_data = json.load(f)    
    return retrain_data
    
def plot_confusion_matrix(confusion_matrix, labels):
    confusion_matrix= np.array(confusion_matrix)

    df_cm = pd.DataFrame(confusion_matrix, index = labels, columns = labels)
    plt.figure(figsize = (7,6))
    sns.heatmap(confusion_matrix, annot=True, cmap='Blues', cbar=False, fmt='g')
    plt.title('Confusion matrix - Retrained model')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
        
def test_acc_mean_std(experiment_path, retrain_file_name):
    retrain_data = load_retrain_results(experiment_path, retrain_file_name)
    test_acc_mean = np.mean([retrain_data[key]['test_accuracy'] for key in retrain_data.keys()])
    test_acc_std = np.std([retrain_data[key]['test_accuracy'] for key in retrain_data.keys()])
    
    return test_acc_mean, test_acc_std    

def agg_results(results_dict):
    # Create an empty dictionary to store the mean and std for each variable
    
    agg_results_dict = {
        "training_losses": [],
        "validation_losses": [],
        "training_accuracies": [],
        "validation_accuracies": [],
        # Add other variables as needed
    }
    # Loop through each dictionary and aggregate the results
    for key in results_dict.keys():
        current_dict = results_dict[key]  # Replace 'results_dicts' with the actual list of dictionaries
        agg_results_dict["training_losses"].append(current_dict["training_losses"])
        agg_results_dict["validation_losses"].append(current_dict["validation_losses"])
        agg_results_dict["training_accuracies"].append(current_dict["training_accuracies"])
        agg_results_dict["validation_accuracies"].append(current_dict["validation_accuracies"])
    
    # Convert the lists to NumPy arrays
    agg_results_dict["training_losses"] = np.array(agg_results_dict["training_losses"])
    agg_results_dict["validation_losses"] = np.array(agg_results_dict["validation_losses"])
    agg_results_dict["training_accuracies"] = np.array(agg_results_dict["training_accuracies"])
    agg_results_dict["validation_accuracies"] = np.array(agg_results_dict["validation_accuracies"])

    # Calculate the mean and std across the first axis (axis=0)
    agg_results_dict["mean_training_losses"] = np.mean(agg_results_dict["training_losses"], axis=0)
    agg_results_dict["std_training_losses"] = np.std(agg_results_dict["training_losses"], axis=0)
    agg_results_dict["mean_validation_losses"] = np.mean(agg_results_dict["validation_losses"], axis=0)
    agg_results_dict["std_validation_losses"] = np.std(agg_results_dict["validation_losses"], axis=0)
    agg_results_dict["mean_training_accuracies"] = np.mean(agg_results_dict["training_accuracies"], axis=0)
    agg_results_dict["std_training_accuracies"] = np.std(agg_results_dict["training_accuracies"], axis=0)
    agg_results_dict["mean_validation_accuracies"] = np.mean(agg_results_dict["validation_accuracies"], axis=0)
    agg_results_dict["std_validation_accuracies"] = np.std(agg_results_dict["validation_accuracies"], axis=0)
    
    return agg_results_dict
        
def plot_training_history(results_dict:dict, params:dict=None, retrain:bool=False, title:str=''):
    """ Plot the training history of a model.
    
    Args:
        results_dict: (dict) dictionary with the training history.
    """
    num_keys = len(results_dict.keys())
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    if retrain:
        if num_keys > 1:
            keys = list(results_dict.keys())
            total_epochs = len(results_dict[keys[0]]['training_losses'])
            epochs = range(1, total_epochs + 1)
            test_acc_mean = np.mean([results_dict[key]['test_accuracy'] for key in results_dict.keys()])
            test_acc_std = np.std([results_dict[key]['test_accuracy'] for key in results_dict.keys()])
            agg_results_dict = agg_results(results_dict)
            ax[0].plot(epochs, agg_results_dict["mean_training_losses"], label='Training', color='blue')
            ax[0].fill_between(epochs, 
                                agg_results_dict["mean_training_losses"] - agg_results_dict["std_training_losses"], 
                                agg_results_dict["mean_training_losses"] + agg_results_dict["std_training_losses"], 
                                color='blue', alpha=0.2)
            ax[0].plot(epochs, agg_results_dict["mean_validation_losses"], label='Validation', color='red')
            ax[0].fill_between(epochs, 
                                agg_results_dict["mean_validation_losses"] - agg_results_dict["std_validation_losses"], 
                                agg_results_dict["mean_validation_losses"] + agg_results_dict["std_validation_losses"], 
                                color='red', alpha=0.2)
            ax[0].set_title('Loss')
            ax[0].set_xlabel('Epochs')
            ax[0].set_ylabel('Loss')
            ax[0].legend(fontsize=12)
            ax[0].grid(True)
            ax[0].set_xlim([1, total_epochs])
            ax[0].set_ylim([0, 1.5])
            
            ax[1].plot(epochs, agg_results_dict["mean_training_accuracies"], label='Training', color='blue')
            ax[1].fill_between(epochs, 
                                agg_results_dict["mean_training_accuracies"] - agg_results_dict["std_training_accuracies"], 
                                agg_results_dict["mean_training_accuracies"] + agg_results_dict["std_training_accuracies"], 
                                color='blue', alpha=0.2)
            ax[1].plot(epochs, agg_results_dict["mean_validation_accuracies"], label='Validation', color='red')
            ax[1].fill_between(epochs, 
                                agg_results_dict["mean_validation_accuracies"] - agg_results_dict["std_validation_accuracies"], 
                                agg_results_dict["mean_validation_accuracies"] + agg_results_dict["std_validation_accuracies"], 
                                color='red', alpha=0.2)
            
            ax[1].axhline(y=test_acc_mean, color='green', linestyle='--', label='Test Accuracy')
            ax[1].text(epochs[-2], test_acc_mean+1, f'{test_acc_mean:.2f} ± {test_acc_std:.2f}', ha='right', va='center', color='black', fontsize=14)
            
            ax[1].set_title('Accuracy')
            ax[1].set_xlabel('Epochs')
            ax[1].set_ylabel('Accuracy')
            ax[1].legend(loc='lower right', fontsize=14)
            ax[1].grid(True)
            ax[1].set_xlim([1, total_epochs])
            # add plt title
            plt.suptitle(f'Training History: {title}', fontsize=16)
            plt.show()
        else:
            results_dict = results_dict[list(results_dict.keys())[0]]
            epochs = range(1, len(results_dict['training_losses']) + 1)
            ax[0].plot(epochs, results_dict["training_losses"], 'b', label='Training loss')
            ax[0].plot(epochs, results_dict["validation_losses"], 'r', label='Validation loss')
            ax[0].set_title('Loss')
            ax[0].set_xlabel('Epoch')
            ax[0].set_ylabel('Loss')
            ax[0].legend()
            ax[0].grid(True)
            
            ax[1].plot(epochs, results_dict["training_accuracies"], 'b', label='Training Acc')
            ax[1].plot(epochs, results_dict["validation_accuracies"], 'r', label='Validation Acc')
            max_acc, index = max(results_dict["validation_accuracies"]), results_dict["validation_accuracies"].index(max(results_dict["validation_accuracies"]))
            ax[1].plot(index+1, max_acc, 'go', label='Max Acc')
            ax[1].text(index+1, max_acc+0.1, f'{max_acc:.2f}', fontsize=12)
            ax[1].set_title('Accuracy')
            ax[1].set_xlabel('Epoch')
            ax[1].set_ylabel('Accuracy')
            ax[1].legend()
            ax[1].grid(True)
    else:
        epochs = range(1, len(results_dict['training_losses']) + 1)
        eval_starts = params["max_epochs"] - params["epochs_to_eval"]
        epochs_val = range(eval_starts+1, max(epochs)+1)
    
        ax[0].plot(epochs, results_dict["training_losses"], 'b', label='Training loss')
        ax[0].plot(epochs_val, results_dict["validation_losses"], 'r', label='Validation loss')
        ax[0].set_title('Loss')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].legend()
        ax[0].grid(True)
        

        ax[1].plot(epochs, results_dict["training_accuracies"], 'b', label='Training Acc')
        ax[1].plot(epochs_val, results_dict["validation_accuracies"], 'r', label='Validation Acc')
        max_acc, index = max(results_dict["validation_accuracies"]), results_dict["validation_accuracies"].index(max(results_dict["validation_accuracies"]))
        ax[1].plot(index+1, max_acc, 'go', label='Max Acc')
        ax[1].text(index+1, max_acc+0.1, f'{max_acc:.2f}', fontsize=12)
        ax[1].set_title('Accuracy')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy')
        ax[1].legend()
        ax[1].grid(True)
    
    plt.show()

def delete_old_dirs(path, keep_best=False, best_id=''):
    """ Delete directories with old training files (models, checkpoints...). Assumes the
        directories' names start with digits.

    Args:
        path: (str) path to the experiment folder.
        keep_best: (bool) True if user wants to keep files from the best individual.
        best_id: (str) id of the best individual.
    """

    folders = [os.path.join(path, d) for d in os.listdir(path)
                if os.path.isdir(os.path.join(path, d)) and d[0].isdigit()]
    folders.sort(key=natural_key)

    if keep_best and best_id:
        folders = [d for d in folders if os.path.basename(d) != best_id]

    for f in folders:
        rmtree(f)

def check_files(exp_path):
    """ Check if exp_path exists and if it does, check if log_file is valid.

    Args:
        exp_path: (str) path to the experiment folder.
    """
    if not os.path.exists(exp_path):
        raise OSError('User must provide a valid "--experiment_path" to continue '
                        'evolution or to retrain a model.')

    # 1. If there’s a symlink named "best_so_far", use its target
    best_link = os.path.join(exp_path, 'best_so_far')
    if os.path.islink(best_link):
        target = os.readlink(best_link)
        if os.path.isdir(target):
            best_result_folder = target
        else:
            raise ValueError(f'"best_so_far" symlink does not point to a directory: {target}')
    else:
        # 2. Otherwise, find subdirectories whose names start with a digit
        experiment_folders = [f.name for f in os.scandir(exp_path) if f.is_dir()]
        digit_folders = [name for name in experiment_folders if name and name[0].isdigit()]
        if not digit_folders:
            raise ValueError(f'No experiment folders starting with a digit found in: {exp_path}')

        # 3. Define numeric sort key (split on '_' and convert digit parts)
        def numeric_key(s):
            parts = s.split('_')
            return tuple(int(p) for p in parts if p.isdigit())

        best_name = min(digit_folders, key=numeric_key)
        best_result_folder = os.path.join(exp_path, best_name)

    # 4. Validate training_params.txt inside the chosen folder
    params_file = os.path.join(best_result_folder, 'training_params.txt')
    if not os.path.exists(params_file):
        raise OSError('training_params.txt not found!')
    if os.stat(params_file).st_size == 0:
        raise OSError('User must provide an "--experiment_path" with a valid data file to '
                        'continue evolution or to retrain a model.')

    # 5. Validate log_params_evolution.txt at the root of exp_path
    log_file = os.path.join(exp_path, 'log_params_evolution.txt')
    if not os.path.exists(log_file):
        raise OSError('log_params_evolution.txt not found!')
    if os.stat(log_file).st_size == 0:
        raise OSError('User must provide an "--experiment_path" with a valid config_file '
                        'to continue evolution or to retrain a model.')

    return best_result_folder
    
def init_log(log_level, name, file_path=None):
    """ Initialize a logging.Logger with level *log_level* and name *name*.

    Args:
        log_level: (str) one of 'NONE', 'INFO' or 'DEBUG'.
        name: (str) name of the module initiating the logger (will be the logger name).
        file_path: (str) path to the log file. If None, stdout is used.

    Returns:
        logging.Logger object.
    """

    logger = logging.getLogger(name)
        # Eliminar handlers existentes para evitar duplicación
    if logger.hasHandlers():
        logger.handlers.clear()

    if file_path is None:
        handler = logging.StreamHandler()
    else:
        handler = logging.FileHandler(file_path)

    formatter = logging.Formatter('%(levelname)s: %(module)s: %(asctime)s.%(msecs)03d '
                                '- %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if log_level == 'INFO':
        logger.setLevel(logging.INFO)
    elif log_level == 'DEBUG':
        logger.setLevel(logging.DEBUG)

    return logger

def load_evolved_data(experiment_path: str):
        """
        Loads evolved data from the specified experiment path.

        Parameters:
        - experiment_path (str): The path to the experiment folder containing evolved data.

        Returns:
        None

        This method reads the evolved data from the best-performing experiment folder within the specified path.
        It extracts information such as neural network details, generation, and individual from the 'training_params.txt' file.

        If the data is in an old format (generation and individual not specified in 'training_params.txt'),
        it attempts to extract them from the folder name using a regular expression.

        The extracted information is stored in the 'evolved_params' attribute of the class.

        Note: This method assumes a specific folder and file structure for evolved data.
        """

        best_so_far_link = os.path.join(experiment_path, 'best_so_far')
        
        if os.path.islink(best_so_far_link):
            best_result_folder = os.readlink(best_so_far_link)
        else:
            experiment_folders = [f.name for f in os.scandir(experiment_path) if f.is_dir()]
            best_result_folder = [name for name in experiment_folders if name[0].isdigit()]
            best_result_folder = os.path.join(experiment_path, best_result_folder[0])
            
        with open(os.path.join(best_result_folder, 'training_params.txt'), 'r') as file:
                best_individual_info = yaml.safe_load(file)
        net_list = best_individual_info.get('net_list', [])
        generation = best_individual_info.get('generation', 0)
        individual = best_individual_info.get('individual', 0)
        best_acc = best_individual_info.get('best_accuracy', 0.0)
        
        if generation == 0 and individual == 0: # only for old format
                matches = re.search(r'(\d+)_(\d+)$', best_result_folder)
                generation = int(matches.group(1))
                individual = int(matches.group(2))

        return {'net': net_list, 'generation': generation, 'individual': individual, 'best_accuracy': best_acc}
    
def load_retrain_results(experiment_path, retrain_file_name='retrain_results_F13_multistep.txt'):
    """
    Load and identify the best retrained model from a JSON results file, then
    return the directory path, best model path, and its network definition.

    Args:
        experiment_path (str):
            The path to the experiment folder containing retraining results.
        retrain_file_name (str, optional):
            The name of the JSON file that stores retrain results 
            (keys map to experiment runs, values include test metrics).
            Defaults to 'retrain_results_F13_multistep.txt'.

    Returns:
        dict:
            A dictionary with:
                - 'net': (list) the network layer definitions from the best run.
                - 'retrain_path': (str) the folder where the best retraining logs 
                and files are stored.
                - 'best_model_path': (str) the file path to the best model checkpoint 
                (`best_model.pth`) in the best retraining folder.

    Raises:
        FileNotFoundError:
            If the determined best retrain folder does not exist, or if the JSON results file 
            or `retraining_params.txt` file are missing or unreadable.
    """
    file_path = os.path.join(experiment_path, retrain_file_name)
    with open(file_path, 'r') as f:
        retrain_data = json.load(f)
        
    # Determine the key with the highest test accuracy
    best_key = max(retrain_data, key=lambda x: retrain_data[x]['test_accuracy'])
    
    # Convert key naming (e.g., "multistep_F13_retrain_1" -> "retrain_F13_1")
    parts = best_key.split("_")
    best_key = f"{parts[2]}_{parts[1]}_{parts[3]}"
    
    # Construct path to the folder for the best retraining run
    retrain_path = os.path.join(experiment_path, best_key)
    if not os.path.exists(retrain_path):
        raise FileNotFoundError(f"Could not find the retrain folder at {retrain_path}")
    
    # Load retraining params (YAML) within the best retraining folder
    with open(os.path.join(retrain_path, 'retraining_params.txt'), 'r') as file:
        best_retrain_info = yaml.safe_load(file)
    
    net_list = best_retrain_info.get('net_list', [])
    
    # Build the path to the best model file
    best_model_path = os.path.join(retrain_path, 'best_model.pth')
        
    return {'net': net_list, 'retrain_path': retrain_path, 'best_model_path': best_model_path}

    
def load_log_params_evolution(experiment_path: str):
    """
    Loads the log parameters for the evolution process from the specified experiment path.

    Parameters:
    - experiment_path (str): The path to the experiment folder containing evolved data.

    Returns:
    dict: A dictionary containing the log parameters for the evolution process.

    This method reads the log parameters for the evolution process from the
    'log_params_evolution.txt' file. These typically include:
        - train_spec  (dict)
        - QNAS_spec   (dict)
        - fn_dict     (dict)
    among other possible keys like population size, generations, mutation rate, etc.
    """

    log_file = os.path.join(experiment_path, 'log_params_evolution.txt')
    if not os.path.isfile(log_file):
        raise FileNotFoundError(f"Could not find log_params_evolution.txt at {log_file}")

    with open(log_file, 'r') as file:
        log_params = yaml.safe_load(file)
    
    # Extract the subsets you need: train, QNAS, fn_dict
    train_spec = dict(log_params['train'])
    QNAS_spec = dict(log_params['QNAS'])    
    fn_dict = log_params['fn_dict']

    # Return them together in a dictionary (you can rename or restructure as you prefer):
    return {
        'train_spec': train_spec,
        'QNAS_spec': QNAS_spec,
        'fn_dict': fn_dict
    }
    
def calculate_time(start_time, elapse_time,current_gen:int=0, max_generations:int=300, end_evol = True):
    """
    Calculate the elapsed time and the estimated remaining time in the evolution process.

    Parameters:
    start_time (int): The start time of the evolution process.
    elapse_time (int): The current time in the evolution process.
    current_gen (int): The current generation number. Default is 0.
    max_generations (int): The maximum number of generations. Default is 300.
    end_evol (bool): If True, only calculate the elapsed time. If False, also calculate the estimated remaining time. Default is True.

    Returns:
    tuple: If end_evol is True, returns a tuple (hours, minutes) representing the elapsed time.
        If end_evol is False, returns a tuple (hours, minutes, remaining_total_hours, remaining_total_minutes) representing the elapsed time and the estimated remaining time.
    """
    
    total_time = elapse_time - start_time
    hours = int(total_time / 3600)
    minutes = int((total_time - hours * 3600) / 60)
    
    if end_evol:
        return hours, minutes
    else:
        avg_time_per_gen = total_time / current_gen if current_gen != 0 else 0
        remaining_total_time = avg_time_per_gen * (max_generations - current_gen)
        remaining_total_hours = int(remaining_total_time / 3600)
        remaining_total_minutes = int((remaining_total_time - remaining_total_hours * 3600) / 60)
        
        return hours, minutes, remaining_total_hours, remaining_total_minutes
    
def download_dataset(params: dict):
    """
    Downloads the specified dataset if it is not already available locally.

    Parameters:
    - params (dict): A dictionary containing the parameters for the dataset.
        - 'data_path' (str): The path where the dataset should be stored.
        - 'dataset' (str): The name of the dataset to be downloaded.

    If the dataset directory specified by 'data_path' does not exist, it will be created, 
    and the dataset will be downloaded. The function supports downloading datasets from 
    torchvision and MedMNIST. If the dataset already exists, it will print a message and 
    skip the download.

    Raises:
    - ValueError: If the dataset is not found in torchvision.datasets or MedMNIST INFO.
    """
    data_path = params['data_path']
    dataset_name = params['dataset'].lower()

    download_status = not os.path.exists(data_path)
    
    if download_status:
        os.makedirs(data_path)

        if hasattr(torchvision.datasets, dataset_name.upper()):
            dataset_family = "pytorch"
            dataset_class = getattr(torchvision.datasets, dataset_name.upper())
            dataset_class(data_path, download=True, transform=ToTensor())
        elif dataset_name in INFO:
            dataset_family = "medmnist"
            general_info = INFO[dataset_name]
            dataset_class = getattr(medmnist, general_info['python_class'])
            dataset_class(root=data_path, split='train', download=True, transform=ToTensor(), as_rgb=True)
        else:
            raise ValueError(f"Dataset class {dataset_name} not found in torchvision.datasets or available_datasets.")
        return False
    else:
        return True

dataset_cache = {}

def _validate_dataset_info(dataset_info: dict, dataset_name: str):
    if not isinstance(dataset_info, dict):
        raise ValueError(f"Dataset info for '{dataset_name}' must be a dict, got {type(dataset_info)}")
    missing = [k for k in ("num_classes", "task", "shape") if k not in dataset_info]
    if missing:
        raise KeyError(f"Dataset info for '{dataset_name}' missing keys: {missing}")
    shape = dataset_info["shape"]
    if not (isinstance(shape, (list, tuple)) and len(shape) == 3):
        raise ValueError(f"'shape' must be [C, H, W] for '{dataset_name}', got: {shape}")

def setup_dataset_info(params):
    """
    Update 'params' with dataset-specific information, prioritizing YAML configs.

    Expects in params:
        - 'dataset' (str)
        - 'batch_size' (int)
        - 'config_path_dataset' (str) -> preferred YAML path, e.g. 'configs/cifar10.yaml'
        OR
        - 'data_path' (str) with an existing 'data_info.txt' (written by the loader)

    Sets:
        - 'num_classes' (int)
        - 'task' (str)
        - 'input_shape' ([B, C, H, W])
    """
    if "dataset" not in params:
        raise KeyError("params['dataset'] is required")
    if "batch_size" not in params:
        raise KeyError("params['batch_size'] is required")

    dataset_name = str(params["dataset"]).lower()

    # Use cache unless the caller explicitly asks to refresh
    if dataset_name in dataset_cache and not params.get("force_reload_dataset_info", False):
        dataset_info = dataset_cache[dataset_name]
    else:
        # Prefer YAML config
        cfg_path = params.get("config_path_dataset")
        if cfg_path and os.path.isfile(cfg_path):
            dataset_info = load_yaml(cfg_path)
        else:
            # Fallback to info file written by the loader
            data_info_path = os.path.join(params.get("data_path", ""), "data_info.txt")
            if not os.path.isfile(data_info_path):
                raise FileNotFoundError(
                    "Could not find dataset YAML at params['config_path_dataset'] "
                    "and fallback 'data_info.txt' does not exist at: "
                    f"{data_info_path}"
                )
            dataset_info = load_yaml(data_info_path)

        _validate_dataset_info(dataset_info, dataset_name)
        dataset_cache[dataset_name] = dataset_info

    # Populate params
    params["num_classes"] = int(dataset_info["num_classes"])
    params["task"] = str(dataset_info["task"])
    c, h, w = [int(x) for x in dataset_info["shape"]]
    params["input_shape"] = [int(params["batch_size"]), c, h, w]
    return params

def get_gpu_memory():
    """
    Retrieve GPU memory usage using GPUtil.
    
    Returns:
    - Used memory in MB.
    """
    gpus = GPUtil.getGPUs()
    if gpus:
        return gpus[0].memoryUsed  # Assuming single-GPU use; modify if using multiple GPUs
    return None


def backup_cache(data, file_path: str = None) -> None:
    """
    Backup (update) the cache of evaluated individuals to a file.
    
    If the backup file exists, load its contents, update them with `data`,
    then write the merged dictionary back to the file. Otherwise, simply
    write `data` to the file.
    
    Args:
        data: dictionary containing evaluated individuals (e.g. self.evaluated).
        file_path: The path to the directory where the backup file is stored.
                If None, you can set a default path.
    """
    if file_path is None:
        file_path = os.getcwd()  # or some default directory
    file_name = os.path.join(file_path, "cache_backup.pkl")
    
    # Load existing cache if it exists
    if os.path.exists(file_name):
        with open(file_name, "rb") as f:
            existing_data = load(f)
        # If both existing_data and data are dictionaries, update the existing data
        if isinstance(existing_data, dict) and isinstance(data, dict):
            existing_data.update(data)
            combined_data = existing_data
        else:
            combined_data = data
    else:
        combined_data = data
    
    with open(file_name, "wb") as f:
        dump(combined_data, f, protocol=HIGHEST_PROTOCOL)

def load_cache(file_path: str) -> Dict:
    """
    Load a cache backup from file into self.evaluated.

    Args:
        file_path: The path to the backup file.
    """
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = load(f)
    else:
        print(f"Cache backup file {file_path} not found. Starting with empty cache.")
        data = {}
    return data


def delete_old_dirs_v2(experiment_path: str,generation: int,keep_ids: List[str],is_snapshot_gen: bool = False,
                    results_subdir: str = "results",archive_subdir: str = "archive",snapshots_subdir: str = "snapshots",
                    link_name: str = "best_so_far",logger: Optional[logging.Logger] = None) -> None:
    """
    Manages experiment artifacts by moving results, taking snapshots,
    pruning the archive, and updating a symlink.
    
    If `is_snapshot_gen` is True, this function will also copy the current
    set of `keep_ids` to a permanent snapshot directory for that generation.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    base = experiment_path
    results_dir = os.path.join(base, results_subdir, f"gen_{generation}")
    archive_dir = os.path.join(base, archive_subdir)
    os.makedirs(archive_dir, exist_ok=True)

    # 1) Move any new results from the temporary 'results' dir to the 'archive' dir
    if os.path.isdir(results_dir):
        for kid in keep_ids:
            src = os.path.join(results_dir, kid)
            dst = os.path.join(archive_dir, kid)
            if os.path.isdir(src) and not os.path.isdir(dst):
                try:
                    shutil.move(src, dst)
                    logger.debug(f"Archived {src} -> {dst}")
                except Exception as e:
                    logger.warning(f"Error moving {src} -> {dst}: {e}")
        try:
            shutil.rmtree(results_dir)
            logger.debug(f"Removed temp dir {results_dir}")
        except Exception as e:
            logger.warning(f"Could not remove temp dir {results_dir}: {e}")

    # --- NEW: Take a snapshot if this is a snapshot generation ---
    if is_snapshot_gen:
        snapshot_gen_dir = os.path.join(base, snapshots_subdir, f"gen_{generation}")
        os.makedirs(snapshot_gen_dir, exist_ok=True)
        logger.info(f"Taking snapshot for generation {generation} -> {snapshot_gen_dir}")
        
        for kid in keep_ids:
            src = os.path.join(archive_dir, kid)
            dst = os.path.join(snapshot_gen_dir, kid)
            if os.path.isdir(src) and not os.path.isdir(dst):
                try:
                    shutil.copytree(src, dst)
                    logger.debug(f"Copied {src} to snapshot {dst}")
                except Exception as e:
                    logger.warning(f"Error copying snapshot for {kid}: {e}")

    # 2) PRUNING: Always prune the main archive to keep it clean.
    #    This removes any models from 'archive/' that are no longer in the current Pareto front.
    for folder in os.listdir(archive_dir):
        if folder not in keep_ids:
            path = os.path.join(archive_dir, folder)
            if os.path.isdir(path):
                try:
                    shutil.rmtree(path)
                    logger.debug(f"Pruned old model from archive: {path}")
                except Exception as e:
                    logger.warning(f"Error pruning {path}: {e}")

    # 3) ACTUALIZAR SYMLINK al primer keep_id
    if not keep_ids:
        logger.error("keep_ids is empty, cannot link best_so_far")
        return

    best = keep_ids[0]
    target = os.path.join(archive_dir, best)
    linkpath = os.path.join(base, link_name)

    # Remove old link or file
    try:
        if os.path.islink(linkpath) or os.path.exists(linkpath):
            os.unlink(linkpath)
    except Exception as e:
        logger.warning(f"Could not remove old symlink {linkpath}: {e}")

    # Create new symlink
    try:
        # Check if target exists before creating a link to it
        if os.path.isdir(target):
            os.symlink(target, linkpath)
            logger.info(f"Updated {link_name} -> {target}")
        else:
            logger.warning(f"Target for symlink {target} does not exist. Cannot create link.")
    except Exception as e:
        logger.error(f"Error creating symlink {linkpath} -> {target}: {e}")
        
def compute_hypervolume_mixed(front_raw: np.ndarray, ref_point=None) -> float:
    """
    Compute hypervolume for a 3-objective Pareto front where:
        - front_raw[:, 0] = accuracy (to be maximized)
        - front_raw[:, 1] = num_parameters (to be minimized)
        - front_raw[:, 2] = inference_time (to be minimized)
    We first convert everything into minimization form by flipping accuracy → -accuracy,
    then build a reference point slightly above the “worst” in each dimension,
    and finally call pymoo’s Hypervolume on that minimization front.
    Args:
        front_raw (np.ndarray): shape=(N, 3) with columns [acc, params, time].
        ref_point (np.ndarray): shape=(3,) with the reference point for hypervolume calculation.
    Returns:
        float: the hypervolume (in the original mixed‐obj space).
    """
    if front_raw is None or len(front_raw) == 0:
        return 0.0
    f = np.array(front_raw, dtype=float, copy=True)
    f[:, 0] = -f[:, 0]  # flip accuracy to minimization
    # Choose a safe reference point (must be worse than all points for minimization)
    if ref_point is None:
        rp = np.max(f, axis=0) + 1e-6
    else:
        rp = np.asarray(ref_point, dtype=float)
    return float(Hypervolume(ref_point=rp).do(f))

def load_pareto_history(filepath="pareto_history.pkl"):
    """
    Carga el archivo pickle que contiene fronts_history.
    Devuelve un dict: {generacion: {nivel_frente: [registros...]}}
    """
    with open(filepath, "rb") as f:
        history = pkl.load(f)
    return history

def plot_hypervolume_over_epochs(main_path, experiment_pattern="exp1_repeat"):

    folder_list = [f for f in os.listdir(main_path) if experiment_pattern in f]
    print(f"Found {len(folder_list)} folders matching the pattern '{experiment_pattern}'.")
    plt.figure(figsize=(10, 6))
    num_plots = 0

    for folder in folder_list:
        history_path = os.path.join(main_path, folder, "pareto_history.pkl")
        if not os.path.exists(history_path):
            continue

        with open(history_path, "rb") as pf:
            history = pkl.load(pf)

        generations = sorted(history.keys())
        hypervolumes = [history[gen].get("hypervolume", 0.0) for gen in generations]

        if generations and any(hypervolumes):
            plt.plot(generations, hypervolumes, marker='o', label=folder)
            num_plots += 1

    plt.title("Hypervolume over Generations for Each Experiment")
    plt.xlabel("Generation")
    plt.ylabel("Hypervolume")
    if num_plots > 0:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def _get_hypervolume_stats(path_or_pattern):
    """
    Helper function to load hypervolume data. It supports three modes:
    1. Path to a single run folder.
    2. Path to a main directory containing multiple run sub-folders.
    3. Path with a prefix to match specific run folders (e.g., "results/exp_").
    """
    run_paths_to_check = []

    # Mode 3: Path is a prefix pattern
    # This is likely if the path itself doesn't exist but its parent directory does.
    if not os.path.exists(path_or_pattern) and os.path.isdir(os.path.dirname(path_or_pattern)):
        parent_dir, prefix = os.path.split(path_or_pattern)
        # Handle case where parent_dir is empty (relative path)
        if not parent_dir:
            parent_dir = '.'
        print(f"Interpreting '{path_or_pattern}' as a prefix pattern.")
        run_paths_to_check = [
            f.path for f in os.scandir(parent_dir)
            if f.is_dir() and f.name.startswith(prefix)
        ]
    # If the path exists, handle Modes 1 and 2
    elif os.path.exists(path_or_pattern):
        # Mode 1: Path is a single run folder
        single_run_pkl = os.path.join(path_or_pattern, "pareto_history.pkl")
        if os.path.isfile(single_run_pkl):
            print(f"Interpreting '{path_or_pattern}' as a single run.")
            run_paths_to_check.append(path_or_pattern)
        # Mode 2: Path is a main directory containing run folders
        elif os.path.isdir(path_or_pattern):
            subdirs = [f.path for f in os.scandir(path_or_pattern) if f.is_dir()]
            if subdirs:
                print(f"Interpreting '{path_or_pattern}' as a main directory containing {len(subdirs)} run(s).")
                run_paths_to_check = subdirs

    if not run_paths_to_check:
        print(f"Warning: No runs found for path/pattern '{path_or_pattern}'")
        return np.array([]), np.array([]), np.array([])

    hvs_by_gen = defaultdict(list)
    # Process all identified runs from any of the modes
    for run_path in run_paths_to_check:
        history_path = os.path.join(run_path, "pareto_history.pkl")
        if not os.path.exists(history_path):
            continue
        with open(history_path, "rb") as pf:
            history = pkl.load(pf)
        for gen, data in history.items():
            if 'hypervolume' in data and isinstance(data['hypervolume'], (int, float)):
                hvs_by_gen[gen].append(data['hypervolume'])

    if not hvs_by_gen:
        print(f"Warning: No valid hypervolume data could be loaded. Please check your 'pareto_history.pkl' files.")
        return np.array([]), np.array([]), np.array([])

    generations = np.array(sorted(hvs_by_gen.keys()))
    mean_hvs = np.array([np.mean(hvs_by_gen[gen]) for gen in generations])
    std_hvs = np.array([np.std(hvs_by_gen[gen]) for gen in generations])
    return generations, mean_hvs, std_hvs


def plot_hypervolume_comparison(path_exp1, path_exp2, label_exp1="Method 1", label_exp2="Method 2"):
    """
    Compares the hypervolume evolution of two experiments by plotting their
    mean performance and standard deviation across multiple runs.

    Args:
        path_exp1 (str): Directory path for the first experiment. This directory
                        should contain a subfolder for each independent run.
        path_exp2 (str): Directory path for the second experiment.
        label_exp1 (str): Plot label for the first experiment.
        label_exp2 (str): Plot label for the second experiment.
    """
    # Get statistics for the first experiment
    print(f"--- Processing Experiment 1: {label_exp1} ---")
    gens1, means1, stds1 = _get_hypervolume_stats(path_exp1)

    # Get statistics for the second experiment
    print(f"\n--- Processing Experiment 2: {label_exp2} ---")
    gens2, means2, stds2 = _get_hypervolume_stats(path_exp2)

    # Set up the plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot for Experiment 1
    if len(gens1) > 0:
        ax.plot(gens1, means1, label=label_exp1, lw=2.5)
        ax.fill_between(gens1, means1 - stds1, means1 + stds1, alpha=0.2, label=f'{label_exp1} (Std. Dev.)')
    else:
        print(f"Could not plot '{label_exp1}' due to lack of data.")

    # Plot for Experiment 2
    if len(gens2) > 0:
        ax.plot(gens2, means2, label=label_exp2, lw=2.5)
        ax.fill_between(gens2, means2 - stds2, means2 + stds2, alpha=0.2, label=f'{label_exp2} (Std. Dev.)')
    else:
        print(f"Could not plot '{label_exp2}' due to lack of data.")
        
    # Final plot styling
    ax.set_title("Hypervolume Comparison", fontsize=16, weight='bold')
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Hypervolume", fontsize=12)
    
    if len(gens1) > 0 or len(gens2) > 0:
        ax.legend(fontsize=11)
        
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout()
    plt.show()
    
def plot_pareto_evolution(history, dims="3d",
                            x="params", y="inference_time", z="accuracy",
                            width=1200, height=800, y_range=None):
    """
    Plot the evolution of Pareto fronts over generations en 2D o 3D con tooltips
    formateados a 2 decimales y unidad.
    """
    # 1) Flatten the history into a DataFrame, skipping the "hypervolume" key
    rows = []
    for gen, fronts in history.items():
        for level, recs in fronts.items():
            # Skip any non-integer key (e.g. "hypervolume")
            if not isinstance(level, int):
                continue

            for rec in recs:
                rows.append({
                    "generation": gen,
                    "front_level": level,
                    "accuracy": rec["accuracy"],
                    "params": rec["params"],
                    "inference_time": rec["inference_time"]
                })

    df = pd.DataFrame(rows)

    # 2) Build the plot in 3D or 2D
    if dims == "3d":
        fig = px.scatter_3d(
            df,
            x=x, y=y, z=z,
            color="front_level",
            animation_frame="generation",
            width=width, height=height,
            title="Pareto Front Evolution (3D)",
            labels={
                x: "Params (M)",
                y: "Inference Time (µs)",
                z: "Accuracy (%)",
                "front_level": "Front Level",
                "generation": "Generation"
            },
            custom_data=["front_level", "generation"]
        )
        fig.update_traces(
            marker=dict(size=4),
            hovertemplate=(
                "Params: %{x:.2f} M<br>"
                "Inference Time: %{y:.2f} µs<br>"
                "Accuracy: %{z:.2f}%<br>"
                "Front Level: %{customdata[0]}<br>"
                "Generation: %{customdata[1]}<extra></extra>"
            )
        )
        fig.update_layout(
            scene=dict(
                yaxis=dict(range=[df[y].min() * 0.9, df[y].max() * 1.5],),
            ),
            margin=dict(l=20, r=20, t=50, b=20)
        )

    else:
        # 2D scatter
        fig = px.scatter(
            df,
            x=x, y=y,
            color="front_level",
            animation_frame="generation",
            width=width, height=height,
            title="Pareto Front Evolution (2D)",
            labels={
                x: "Params (M)",
                y: "Inference Time (µs)",
                "front_level": "Front Level",
                "generation": "Generation"
            },
            custom_data=["front_level", "generation"]
        )
        fig.update_traces(
            marker=dict(size=6),
            hovertemplate=(
                "Params: %{x:.2f} M<br>"
                "Inference Time: %{y:.2f} µs<br>"
                "Front Level: %{customdata[0]}<br>"
                "Generation: %{customdata[1]}<extra></extra>"
            )
        )
        if y_range is not None:
            fig.update_layout(yaxis=dict(range=y_range))
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))

    fig.show()
    
def load_data_for_pareto(file_path):
    """
    Loads and processes data from a single large JSON results file and 
    formats it for the plot_pareto_evolution function.

    This version correctly reads the entire file as a single JSON object.

    Args:
        file_path (str): The path to the input JSON file.

    Returns:
        dict: A dictionary formatted for the plot_pareto_evolution function.
    """
    # Open and load the entire file as a single JSON object
    with open(file_path, 'r') as f:
        full_data = json.load(f)

    # Since 'generation' and 'front_level' are not in the source data,
    # we place all results into a single generation (0) and front (0) for plotting.
    pareto_front_records = []

    # Iterate through each main key (e.g., "0_17", "0_2") in the JSON data
    for key, retrain_data in full_data.items():
        # Lists to store metrics from each retrain trial for averaging
        accuracies = []
        params_list = []
        inference_times = []

        # Iterate through each retrain instance (e.g., "retrain_1", "retrain_2")
        for retrain_key, metrics in retrain_data.items():
            # Ensure the retrain entry has the necessary metrics
            if "test_accuracy" in metrics and "total_params" in metrics and "cuda_inference_time" in metrics:
                accuracies.append(metrics['test_accuracy'])
                params_list.append(metrics['total_params'])
                inference_times.append(metrics['cuda_inference_time'])

        # If we found any valid retrain data, calculate the means
        if accuracies:
            mean_accuracy = statistics.mean(accuracies)
            mean_params = statistics.mean(params_list)
            mean_inference_time = statistics.mean(inference_times)
            
            pareto_front_records.append({
                "accuracy": mean_accuracy,
                # Convert params to millions (M) for a more readable plot scale
                "params": mean_params / 1_000_000, 
                "inference_time": mean_inference_time
            })

    # Structure the data into the nested dictionary format expected by the plotting function
    history = {
        0: {  # Generation 0
            0: pareto_front_records  # Front Level 0
        }
    }
    return history

def load_history_from_json(file_path: str) -> dict:
    """Loads a history database from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            history_from_json = json.load(f)
        history_db = {tuple(map(int, k.split(','))): v for k, v in history_from_json.items()}
        print(f"Successfully loaded {len(history_db)} architectures from {file_path}")
        return history_db
    except FileNotFoundError:
        print(f"History file not found at {file_path}. Starting with an empty database.")
        return {}

def save_history_to_json(history: dict, file_path: str):
    """
    Saves the current state of the history database to a JSON file.

        history: dict: The history database to save.
        file_path: str: The path to the JSON file where the history will be saved.
    """
    # Convert tuple keys to comma-separated strings to make them JSON-compatible.
    history_for_json = {",".join(map(str, k)): v for k, v in history.items()}
    
    try:
        # Write to a temporary file first to prevent data corruption if the script crashes mid-write
        temp_file_path = file_path + ".tmp"
        with open(temp_file_path, 'w') as f:
            json.dump(history_for_json, f, indent=4)
        # If write is successful, rename the temporary file to the final name
        os.replace(temp_file_path, file_path)
    except IOError as e:
        print("Failed to save history file: %s", e)    