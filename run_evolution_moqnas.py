import argparse
import os

from moqnas import MOQNAS
import qnas_config as cfg
import evaluation
from util import check_files, init_log, download_dataset

def main(**args):
    logger = init_log(args['log_level'], name=__name__)

    if not os.path.exists(args['experiment_path']):
        logger.info(f"Creating {args['experiment_path']} ...")
        os.makedirs(args['experiment_path'])

    # Start new evolution or continue previous one
    if not args['continue_path']:
        phase = 'evolution'
    else:
        phase = 'continue_evolution'
        logger.info(f"Continuing evolution from: {args['continue_path']}. Checking files ...")
        check_files(args['continue_path'])

    logger.info(f"Loading parameters from {args['config_file']} ...")
    config = cfg.ConfigParameters(args, phase=phase)
    config.get_parameters()
    logger.info(f"Saving parameters for phase {config.phase} ...")
    config.save_params_logfile()

    if config.train_spec['mixed_precision']:
        logger.info("Using mixed precision training ...")

    # Download dataset if needed
    dataset_status = download_dataset(params=config.train_spec)
    status_message = "Dataset was already downloaded." if dataset_status else "Dataset downloaded successfully."
    logger.info(status_message)

    eval_pop = evaluation.EvalPopulation(
        params=config.train_spec,
        fn_dict=config.fn_dict,
        log_level=config.train_spec['log_level']
    )

    moqnas = MOQNAS(
        eval_func=eval_pop,
        experiment_path=config.train_spec['experiment_path'],
        objectives=config.train_spec['objectives'],
        log_file=config.files_spec['log_file'],
        log_level=config.train_spec['log_level'],
        data_file=config.files_spec['data_file']
    )
    
    moqnas.initialize_moqnas(**config.QNAS_spec)

    logger.info("Starting MO-QNAS evolution ...")
    pareto_pop, pareto_fits = moqnas.evolve()
    logger.info("MO-QNAS evolution finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_path', type=str, required=True,
                        help='Directory where logs and model files will be saved.')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the input data.')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cifar10', 'cifar100', 'pathmnist', 'octmnist', 'tissuemnist',
                                'organamnist', 'organcmnist', 'atleta_axial', 'atleta_coronal'],
                        help='Name of the dataset.')
    parser.add_argument('--config_file', type=str, required=True,
                        help='Configuration file name.')
    parser.add_argument('--continue_path', type=str, default='',
                        help='If continuing a previous evolution, specify its directory.')
    parser.add_argument('--log_level', choices=['NONE', 'INFO', 'DEBUG'], default='NONE',
                        help='Logging level.')
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        choices=['RMSProp', 'Adam', 'AdamW', 'SGD'],
                        help='Optimizer to use during training. Default = AdamW.')
    parser.add_argument('--fitness_metric', type=str, default='best_accuracy',
                        choices=['best_accuracy', 'best_loss', 'scalar_multi_objective'],
                        help='Fitness metric for evolution. Default = best_accuracy.')
    parser.add_argument('--data_augmentation', action='store_true',
                        help='Enable data augmentation during training. Default = False.')
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable early stopping in evolution. Default = False.')
    parser.add_argument('--en_pop_crossover', action='store_true',
                        help='Enable population crossover during evolution. Default = False.')
    parser.add_argument('--save_checkpoints_epochs', type=int, default=5,
                        help='Number of epochs between saving checkpoints. Default = 5.')
    parser.add_argument('--limit_data_value', type=int, default=10000,
                        help='Number of samples to use during evolution and training. Default = 10000.')
    parser.add_argument('--backbone_name', type=str, default='mobilenet_v3_small',
                        choices=['mobilenet_v3_small', 'mobilenet_v3_large', 'mobilenet_v2',
                                'resnet18', 'resnet50'],
                        help='Backbone to use during training. Default = mobilenet_v3_small.')
    parser.add_argument('--network_config', type=str, required=True, default='default',
                        choices=['default', 'dense', 'backbone'],
                        help='Network structure configuration.')

    arguments = parser.parse_args()
    main(**vars(arguments))
