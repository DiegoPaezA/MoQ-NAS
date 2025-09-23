import os
import argparse
from typing import Tuple

from core import config as cfg
from core import evaluation
from utils.helpers import check_files, init_log, download_dataset

from algorithms.qnas.moqnas import MOQNAS
from algorithms.qnas import qnas2 as qnas
from algorithms.ga import base_ga as ga
from algorithms.ga import nsga2, nsga3


def _bootstrap(logger, args) -> Tuple[object, object, str]:
    """Shared setup for all algorithms: config, dataset, EvalPopulation."""
    if not os.path.exists(args['experiment_path']):
        logger.info(f"Creating {args['experiment_path']} ...")
        os.makedirs(args['experiment_path'])

    phase = 'evolution' if not args['continue_path'] else 'continue_evolution'
    if phase == 'continue_evolution':
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

    dataset_status = download_dataset(params=config.train_spec)
    logger.info("Dataset is already downloaded." if dataset_status else "Dataset downloaded successfully.")

    eval_pop = evaluation.EvalPopulation(
        params=config.train_spec,
        fn_dict=config.fn_dict,
        log_level=config.train_spec['log_level']
    )
    return config, eval_pop, phase


def main(**args):
    logger = init_log(args['log_level'], name=__name__)
    config, eval_pop, _ = _bootstrap(logger, args)

    algo = args.get('algo', 'nsga2').lower()
    logger.info(f"Selected algorithm: {algo}")

    # If fn_list wasn’t set via CLI, fall back to config
    if args.get('fn_list') is None:
        args['fn_list'] = config.QNAS_spec.get('fn_list')

    evolve_returns_four = False  # GA returns 4 values; others return 2

    # -------- Instantiate the engine --------
    if algo == 'ga':
        if ga is None:
            raise ImportError("algorithms.ga.base_ga not found.")
        logger.info("Using GA (single-objective).")
        engine = ga.GA(
            eval_pop,
            config.train_spec['experiment_path'],
            objectives=config.train_spec['objectives'],
            log_file=config.files_spec['log_file'],
            log_level=config.train_spec['log_level'],
            data_file=config.files_spec['data_file'],
            use_cache=args['use_cache'],
        )
        evolve_returns_four = True

    elif algo == 'nsga3':
        if nsga3 is None:
            raise ImportError("algorithms.ga.nsga3 not found.")
        logger.info("Using NSGA-III.")
        engine = nsga3.NSGA3(
            eval_pop,
            config.train_spec['experiment_path'],
            objectives=config.train_spec['objectives'],
            log_file=config.files_spec['log_file'],
            log_level=config.train_spec['log_level'],
            data_file=config.files_spec['data_file'],
            use_cache=args['use_cache'],
            ref_divisions=args.get('ref_divisions', None),
        )

    elif algo == 'nsga2':
        if nsga2 is None:
            raise ImportError("algorithms.ga.nsga2 not found.")
        logger.info("Using NSGA-II.")
        engine = nsga2.NSGA2(
            eval_func=eval_pop,
            experiment_path=config.train_spec['experiment_path'],
            objectives=config.train_spec['objectives'],
            log_file=config.files_spec['log_file'],
            log_level=config.train_spec['log_level'],
            data_file=config.files_spec['data_file'],
            use_cache=args['use_cache'],
        )

    elif algo == 'qnas':
        if qnas is None:
            raise ImportError("algorithms.qnas.qnas2 not found.")
        logger.info("Using QNAS.")
        engine = qnas.QNAS(
            eval_pop,
            config.train_spec['experiment_path'],
            objectives=config.train_spec['objectives'],
            log_file=config.files_spec['log_file'],
            log_level=config.train_spec['log_level'],
            data_file=config.files_spec['data_file'],
            use_cache=args['use_cache'],
        )
        # Special initializer for QNAS
        engine.initialize_qnas(**config.QNAS_spec)
        logger.info("Starting QNAS evolution ...")
        engine.evolve()
        logger.info("QNAS evolution finished.")
        return  # QNAS doesn’t return (pop, fits)

    elif algo == 'moqnas':
        if MOQNAS is None:
            raise ImportError("algorithms.qnas.moqnas.MOQNAS not found.")
        logger.info("Using MO-QNAS.")
        engine = MOQNAS(
            eval_func=eval_pop,
            experiment_path=config.train_spec['experiment_path'],
            objectives=config.train_spec['objectives'],
            log_file=config.files_spec['log_file'],
            log_level=config.train_spec['log_level'],
            data_file=config.files_spec['data_file'],
        )
        # Special initializer for MO-QNAS
        engine.initialize_moqnas(**config.QNAS_spec)
        logger.info("Starting MO-QNAS evolution ...")
        engine.evolve()
        logger.info("MO-QNAS evolution finished.")
        return

    else:
        raise ValueError("Unknown --algo. Choose from: ga, nsga2, nsga3, qnas, moqnas.")

    # -------- Shared GA-style init (GA / NSGA-II / NSGA-III) --------
    engine.initialize_ga(
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

    # -------- Run evolution --------
    logger.info("Starting evolution ...")
    if evolve_returns_four:
        population, fitnesses, best_fitness, best_id = engine.evolve()
        logger.info("Evolution finished.")
        logger.info(f"Best fitness: {best_fitness}")
        logger.info(f"Best individual (generation, index): {best_id}")
    else:
        population, fitnesses = engine.evolve()
        logger.info("Evolution finished.")
        if population is not None and fitnesses is not None:
            for i, (ind, fit) in enumerate(zip(population, fitnesses)):
                chrom = ind.tolist() if hasattr(ind, "tolist") else ind
                print(f"  Ind {i}: chrom={chrom}  →  (acc={fit[0]:.3f}, params={fit[1]:.0f}, time={fit[2]:.4f})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_path', type=str, required=True,
                        help='Directory where to write logs and model files.')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to input data.')
    parser.add_argument('--config_path_dataset', type=str, required=True,
                        help='Path to dataset config file (YAML).')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cifar10', 'cifar100', 'pathmnist', 'octmnist', 'tissuemnist',
                                'organamnist', 'organcmnist', 'atleta_axial', 'atleta_coronal'],
                        help='Dataset name.')
    parser.add_argument('--config_file', type=str, required=True,
                        help='Configuration file name.')
    parser.add_argument('--continue_path', type=str, default='',
                        help='If resuming a previous evolution, point to its experiment path.')
    parser.add_argument('--log_level', choices=['NONE', 'INFO', 'DEBUG'], default='NONE',
                        help='Logging verbosity.')

    # Training / pipeline toggles (kept from your scripts)
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        choices=['RMSProp', 'Adam', 'AdamW', 'SGD'])
    parser.add_argument('--fitness_metric', type=str, default='best_accuracy',
                        choices=['best_accuracy', 'best_loss', 'scalar_multi_objective'])
    parser.add_argument('--data_augmentation', action='store_true')
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--en_pop_crossover', action='store_true')
    parser.add_argument('--save_checkpoints_epochs', type=int, default=5)
    parser.add_argument('--limit_data_value', type=int, default=10000)
    parser.add_argument('--backbone_name', type=str, default='mobilenet_v3_small',
                        choices=['mobilenet_v3_small', 'mobilenet_v3_large',
                                'mobilenet_v2', 'resnet18', 'resnet50'])
    parser.add_argument('--network_config', type=str, default='default',
                        choices=['default', 'dense', 'backbone'])

    # Algorithm selector
    parser.add_argument('--algo', type=str, default='nsga2',
                        choices=['ga', 'nsga2', 'nsga3', 'qnas', 'moqnas'],
                        help='Which evolutionary algorithm to run.')

    # GA/NSGA params
    parser.add_argument('--population_size', type=int, default=20)
    parser.add_argument('--num_generations', type=int, default=50)
    parser.add_argument('--max_num_nodes', type=int, default=20)
    parser.add_argument('--fn_list', nargs='+', type=str, default=None,
                        help='Layer/op names used to decode chromosomes (fallback to config).')
    parser.add_argument('--crossover_rate', type=float, default=0.9)
    parser.add_argument('--mutation_rate', type=float, default=0.05)
    parser.add_argument('--elitism', action='store_true')
    parser.add_argument('--patience', type=int, default=60)

    # NSGA-III specific
    parser.add_argument('--ref_divisions', type=int, default=None,
                        help='NSGA-III lattice divisions p (auto if None).')

    # QNAS/MO-QNAS extras (kept for compatibility)
    parser.add_argument('--elite_mode', type=str, default='global_k',
                        choices=['single', 'global_k', 'bootstrap_k', 'old', 'moead_topk'])
    parser.add_argument('--ref_dir_method', type=str, default='das-dennis',
                        choices=['das-dennis', 'dirichlet'])
    parser.add_argument('--no-truncate-after-noop', action='store_true', dest='truncate_after_noop',
                        help='Disable truncating architectures after the first no-op.')
    parser.add_argument('--no-avoid-consecutive-pool', action='store_true', dest='avoid_consecutive_pool',
                        help='Disable the rule preventing consecutive pooling layers.')
    parser.add_argument('--no-enforce-noop-in-update', action='store_true', dest='enforce_noop_in_update',
                        help='Disable enforcing no-op rules during the quantum update.')

    # Cache
    parser.add_argument('--use_cache', action='store_true', default=False,
                        help='Use cached evaluations to speed up runs.')

    args = parser.parse_args()
    main(**vars(args))