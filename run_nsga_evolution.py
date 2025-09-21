import sys
import os
import argparse

from algorithms.ga import nsga2, nsga3
from core import config as cfg
from core import evaluation
from utils.helpers import check_files, init_log, download_dataset


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

    if args['early_stopping']:
        logger.info(f"Early stopping is enabled. Patience: {args['patience']} generations.")

    if args["use_cache"]:
        logger.info("Using cached evaluations to speed up runs.")

    logger.info(f"Getting parameters from {args['config_file']} ...")
    config = cfg.ConfigParameters(args, phase=phase)
    config.get_parameters()
    logger.info(f"Saving parameters for {config.phase} phase ...")
    config.save_params_logfile()

    if config.train_spec.get('mixed_precision', False):
        logger.info("Using mixed precision training ...")

    # Download dataset (no-op if already present)
    dataset_status = download_dataset(params=config.train_spec)
    status_message = "Dataset is already downloaded." if dataset_status else "Dataset downloaded successfully."
    logger.info(status_message)

    eval_pop = evaluation.EvalPopulation(params=config.train_spec,
                                         fn_dict=config.fn_dict,
                                         log_level=config.train_spec['log_level'])

    # --- Choose MOEA: NSGA-II (default) or NSGA-III ---
    algo = args.get('mo_algo', 'nsga2').lower()
    if algo == 'nsga3':
        if nsga3 is None:
            raise ImportError("nsga3.py not found or failed to import. Place nsga3.py next to nsga2.py.")
        logger.info("Using NSGA-III for multi-objective evolution.")
        nsga_cnn = nsga3.NSGA3(
            eval_pop,
            config.train_spec['experiment_path'],
            objectives=config.train_spec['objectives'],
            log_file=config.files_spec['log_file'],
            log_level=config.train_spec['log_level'],
            data_file=config.files_spec['data_file'],
            use_cache=args['use_cache'],
            ref_divisions=args.get('ref_divisions', None),
        )
    else:
        if algo != 'nsga2':
            logger.warning(f"Unknown mo_algo '{algo}', falling back to NSGA-II.")
        logger.info("Using NSGA-II for multi-objective evolution.")
        nsga_cnn = nsga2.NSGA2(
            eval_func=eval_pop,
            experiment_path=config.train_spec['experiment_path'],
            objectives=config.train_spec['objectives'],
            log_file=config.files_spec['log_file'],
            log_level=config.train_spec['log_level'],
            data_file=config.files_spec['data_file'],
            use_cache=args['use_cache'],
        )

    # If fn_list not passed on CLI, fall back to config
    if args.get('fn_list') is None:
        args['fn_list'] = config.QNAS_spec['fn_list']

    # Initialize GA with all parameters
    nsga_cnn.initialize_ga(
        population_size=args['population_size'],
        num_generations=args['num_generations'],
        max_num_nodes=args['max_num_nodes'],
        fn_list=args['fn_list'],
        crossover_rate=args['crossover_rate'],
        mutation_rate=args['mutation_rate'],
        early_stopping=args['early_stopping'],
        elitism=args['elitism'],
        patience=args['patience'],
        params_ranges=config.QNAS_spec['params_ranges'],
    )

    # Start evolution
    logger.info("Starting evolution.")
    population, fitnesses = nsga_cnn.evolve()
    logger.info("Evolution finished.")
    for i, (ind, fit) in enumerate(zip(population, fitnesses)):
        print(
            f"  Ind {i}: chrom={ind.tolist()}  →  "
            f"(acc={fit[0]:.3f}, params={fit[1]:.0f}, time={fit[2]:.4f})"
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_path', type=str, required=True,
                        help='Directory where to write logs and model files.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to input data.')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name.',
                        choices=['cifar10', 'cifar100', 'pathmnist', 'octmnist', 'tissuemnist',
                                 'organamnist', 'organcmnist', 'atleta_axial', 'atleta_coronal'])
    parser.add_argument('--config_file', type=str, required=True,
                        help='Configuration file name.')
    parser.add_argument('--continue_path', type=str, default='',
                        help='Resume evolution from a previous experiment path (loads parameters).')
    parser.add_argument('--log_level', choices=['NONE', 'INFO', 'DEBUG'], default='NONE',
                        help='Logging information level.')
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        choices=['RMSProp', 'Adam', 'AdamW', 'SGD'],
                        help='Optimizer to be used during training. Default = AdamW.')
    parser.add_argument('--fitness_metric', type=str, default='best_accuracy',
                        choices=['best_accuracy', 'best_loss', 'scalar_multi_objective'],
                        help='Fitness metric when not using multi-objective.')
    parser.add_argument('--data_augmentation', action='store_true',
                        help='Enable data augmentation during training. Default = False.')
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable evolutionary early stopping. Default = False.')
    parser.add_argument('--en_pop_crossover', action='store_true',
                        help='Enable population crossover during evolution. Default = False.')
    parser.add_argument('--save_checkpoints_epochs', type=int, default=5,
                        help='Save model checkpoint every N epochs. Default = 5.')
    parser.add_argument('--limit_data_value', type=int, default=10000,
                        help='Number of samples to use during evolution/training. Default = 10000.')
    parser.add_argument('--backbone_name', type=str, default='mobilenet_v3_small',
                        choices=['mobilenet_v3_small', 'mobilenet_v3_large', 'mobilenet_v2',
                                'resnet18', 'resnet50'],
                        help='Backbone to use. Default = mobilenet_v3_small.')
    parser.add_argument('--network_config', type=str, default='default',
                        choices=['default', 'dense', 'backbone'],
                        help='Network structure configuration.')

    # GA parameters
    parser.add_argument('--population_size', type=int, required=True,
                        help='Number of individuals in the GA population.')
    parser.add_argument('--num_generations', type=int, required=True,
                        help='Maximum number of generations to evolve.')
    parser.add_argument('--max_num_nodes', type=int, required=True,
                        help='Length of each individual chromosome (max nodes).')
    parser.add_argument('--fn_list', nargs='+', type=str, default=None,
                        help='Function names (layer types) to decode chromosomes. '
                            'If omitted, loaded from config.QNAS_spec["fn_list"].')
    parser.add_argument('--crossover_rate', type=float, required=True,
                        help='Probability of crossover (0..1).')
    parser.add_argument('--mutation_rate', type=float, required=True,
                        help='Probability of mutation per gene (0..1).')
    parser.add_argument('--elitism', action='store_true',
                        help='Keep the best individual into the next generation.')
    parser.add_argument('--patience', type=int, default=60,
                        help='Generations with no improvement before early stopping.')

    # Multi-objective toggles
    parser.add_argument('--num_objectives', type=int, default=3,
                        help='Number of objectives (e.g., accuracy, params, time).')
    parser.add_argument('--multi_objective', action='store_true', default=False,
                        help='Enable multi-objective optimization.')
    parser.add_argument('--mo_algo', type=str, default='nsga2',
                        choices=['nsga2', 'nsga3'],
                        help='Multi-objective algorithm to use.')
    parser.add_argument('--ref_divisions', type=int, default=None,
                        help='NSGA-III: lattice divisions p (auto if None).')

    # Eval cache
    parser.add_argument('--use_cache', action='store_true', default=False,
                        help='Use cached evaluations to speed up runs. Default = False.')

    arguments = parser.parse_args()
    main(**vars(arguments))
