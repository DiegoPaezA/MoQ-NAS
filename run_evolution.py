import argparse
import os

import qnas2 as qnas
import qnas_config as cfg
import evaluation
from util import check_files, init_log, download_dataset

def main(**args):
    
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

    if args["use_cache"]:
        logger.info(f"Using cached evaluations to speed up runs.")
        
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
    
    qnas_cnn = qnas.QNAS(eval_pop, config.train_spec['experiment_path'],
                        objectives=config.train_spec['objectives'],
                        log_file=config.files_spec['log_file'],
                        log_level=config.train_spec['log_level'],
                        data_file=config.files_spec['data_file'],
                        use_cache=args['use_cache'])

    qnas_cnn.initialize_qnas(**config.QNAS_spec)
    
    # Start evolution
    logger.info(f"Starting evolution ...")
    qnas_cnn.evolve()
    logger.info(f"Evolution finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_path', type=str, required=True,
                        help='Directory where to write logs and model files.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to input data.')
    parser.add_argument('--config_path_dataset', type=str, required=True,
                        help='Path to dataset config file (YAML).')
    parser.add_argument('--dataset', type=str, required=True,  help='Dataset name.', 
                        choices=['cifar10', 'cifar100', 'pathmnist', 'octmnist', 'tissuemnist', 'organamnist', 'organcmnist', 'atleta_axial', 'atleta_coronal'])
    parser.add_argument('--config_file', type=str, required=True,
                        help='Configuration file name.')
    parser.add_argument('--continue_path', type=str, default='',
                        help='If the user wants to continue a previous evolution, point to '
                            'the corresponding experiment path. Evolution parameters will be '
                            'loaded from this folder.')
    parser.add_argument('--log_level', choices=['NONE', 'INFO', 'DEBUG'], default='NONE',
                        help='Logging information level.')
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['RMSProp', 'Adam', 'AdamW', 'SGD'],
                        help='Optimizer to be used during training. Default = AdamW.')
    parser.add_argument('--fitness_metric', type=str, default='best_accuracy', 
                        choices=['best_accuracy', 'best_loss', 'scalar_multi_objective'],
                        help='Fitness metric to be used during evolution. Default = accuracy.')
    parser.add_argument('--data_augmentation', action='store_true',
                    help='Enable data augmentation during training. Default = False.')
    parser.add_argument('--early_stopping', action='store_true',
                    help='Enable evolutionary early stopping. Default = False.')
    parser.add_argument('--en_pop_crossover', action='store_true',
                    help='Enable population crossover during evolution. Default = False.') 
    parser.add_argument('--save_checkpoints_epochs', type=int, default=5,
                        help='Number of epochs to save the model. Default = 5.')
    parser.add_argument('--limit_data_value', type=int, default=10000,
                        help='Number of samples to be used during evolution and training. Default = 10000.')
    parser.add_argument('--backbone_name', type=str, default='mobilenet_v3_small', 
                        choices=['mobilenet_v3_small', 'mobilenet_v3_large', 'mobilenet_v2', 'resnet18', 'resnet50',],
                        help='Backbone name to be used during training. Default = mobilenet_v3_small.')
    parser.add_argument('--network_config', type=str, required=True,  help='Network structure configuration.', default='default',
                        choices=['default', 'dense', 'backbone'])

    parser.add_argument('--elite_mode', type=str, default='global_k',
                        choices=['single', 'global_k', 'bootstrap_k', 'old'],
                        help='Elite selection mode during evolution. Default = global_k.')


# --- New Network Architecture Rule Flags ---
    parser.add_argument('--no-truncate-after-noop', action='store_true', dest='truncate_after_noop', default=False,
                        help='Disable truncating architectures after the first no-op. (Default: Enabled)')
    parser.add_argument('--no-avoid-consecutive-pool', action='store_true', dest='avoid_consecutive_pool', default=False,
                        help='Disable the rule preventing consecutive pooling layers. (Default: Enabled)')
    parser.add_argument('--no-enforce-noop-in-update', action='store_true', dest='enforce_noop_in_update', default=False,
                        help='Disable enforcing no-op rules during the quantum update. (Default: Enabled)')

    # Use cache for evaluations
    parser.add_argument('--use_cache', action='store_true', default=False,
                        help='Use cached evaluations to speed up runs. Default = False.')
    
    arguments = parser.parse_args()

    main(**vars(arguments))