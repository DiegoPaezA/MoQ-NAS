"""Checkpoint/resume for the quantum-inspired search state (Area 6).

One checkpoint file per experiment (``<experiment_path>/checkpoint.pkl``),
written atomically at every generation boundary — the end of
``go_next_gen``, after the quantum update, archive update and
``save_data`` are complete and before the counter advances — and
overwritten by default (``checkpoint_keep_every: N`` in the train config
additionally keeps an immutable ``checkpoint_gen{g}.pkl`` every N
generations).

The captured state covers everything a faithful continuation needs:

- generation counter and bookkeeping (``total_eval``, best-so-far,
  early-stopping counter, the per-generation update intensity draw);
- the quantum population: PMF tensor (``qpop_net.probabilities``),
  current sampled/elite populations, chromosome length, the
  self-incremented update counter ``u`` (``metrics_logger``) and the
  elite-EMA ``_q_ema`` (``update_strategies``) — the two pieces of
  coupled elite state whose loss would corrupt every post-resume update;
- the continuous population bounds/values (``qpop_params``);
- the classical survivors and the external nondominated archive
  (MO-QNAS attributes, skipped for plain QNAS);
- ALL RNG states (numpy global — the search backbone — plus Python
  ``random`` and torch CPU/CUDA);
- a config block (NQ, M/fn_list, cadences, objectives, plus the
  evaluation fingerprint/precision/seed injected by the entry point)
  validated field-by-field on resume; any mismatch aborts naming the
  differing fields. Reference directions are rebuilt from config (both
  das-dennis and dirichlet are deterministic — the latter uses a fixed
  internal seed), so they are covered by the config comparison.

Resume requires the explicit ``--resume`` flag; without it an existing
checkpoint is ignored and the run starts from generation 0.
"""
import os
import pickle
import random
import tempfile

import numpy as np
import torch

FORMAT_VERSION = 1


def checkpoint_path(engine) -> str:
    return os.path.join(engine.experiment_path, 'checkpoint.pkl')


def _config_block(engine) -> dict:
    qn = engine.qpop_net
    block = {
        'num_quantum_ind': int(qn.num_ind),
        'fn_list': list(qn.chromosome.fn_list),
        'max_generations': int(getattr(engine, 'max_generations', 0)
                               or getattr(engine, 'generations', 0) or 0),
        'update_quantum_gen': int(engine.update_quantum_gen),
        'crossover_frequency': int(getattr(engine, 'crossover_frequency', 0) or 0),
        'objectives': list(engine.objectives),
        'algorithm': type(engine).__name__,
    }
    # Injected by run_all_evolution: evaluation fingerprint, precision, seed.
    block.update(getattr(engine, 'checkpoint_extra', {}) or {})
    return block


def save_checkpoint(engine) -> None:
    """Serialize the full search state at the current generation boundary."""
    qn, qp = engine.qpop_net, engine.qpop_params
    state = {
        'format_version': FORMAT_VERSION,
        'completed_gen': int(engine.current_gen),
        'total_eval': engine.total_eval,
        'best_so_far': engine.best_so_far,
        'last_best_so_far': getattr(engine, 'last_best_so_far', None),
        'best_so_far_id': getattr(engine, 'best_so_far_id', None),
        'early_stopping_counter': getattr(engine, 'early_stopping_counter', 0),
        'random_intensity': getattr(engine, 'random', 0.0),
        'qpop_net': {
            'probabilities': qn.probabilities,
            'current_pop': qn.current_pop,
            'current_pop_objs': getattr(qn, 'current_pop_objs', None),
            'num_genes': qn.chromosome.num_genes,
            'update_counter': qn.logger._update_counter,
            'last_P': qn.logger._last_P,
            'q_ema': qn.update_strategy._q_ema,
        },
        'qpop_params': {
            'lower': qp.lower, 'upper': qp.upper, 'current_pop': qp.current_pop,
        },
        # torch RNG states are stored as numpy arrays: a raw torch.Tensor
        # pickles with non-deterministic metadata (identical content, different
        # bytes), which would make otherwise-equal checkpoints byte-differ.
        'rng': {
            'numpy': np.random.get_state(),
            'python': random.getstate(),
            'torch_cpu': torch.get_rng_state().numpy(),
            'torch_cuda': ([s.numpy() for s in torch.cuda.get_rng_state_all()]
                           if torch.cuda.is_available() else None),
        },
        'config': _config_block(engine),
    }
    # MO-QNAS classical survivors + external archive (absent in plain QNAS).
    if hasattr(engine, 'pareto_global_population'):
        state['moqnas'] = {
            'classical_nets': engine.classical_nets,
            'classical_params': engine.classical_params,
            'classical_ids': list(engine.classical_ids),
            'fits': engine.fits,
            'raw_fits': engine.raw_fits,
            'pareto_global_population': engine.pareto_global_population,
            'pareto_global_fitnesses': engine.pareto_global_fitnesses,
            'pareto_global_params': engine.pareto_global_params,
            'pareto_global_ids': list(engine.pareto_global_ids),
            'fronts_history': engine.fronts_history,
        }

    path = checkpoint_path(engine)
    parent = os.path.dirname(path) or '.'
    fd, tmp = tempfile.mkstemp(prefix='.checkpoint.', suffix='.tmp', dir=parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    keep_every = int(getattr(engine, 'checkpoint_keep_every', 0) or 0)
    if keep_every and engine.current_gen % keep_every == 0:
        import shutil
        shutil.copy2(path, os.path.join(
            engine.experiment_path, f'checkpoint_gen{engine.current_gen}.pkl'))


def load_checkpoint(engine) -> int:
    """Validate and restore a checkpoint into ``engine``.

    Returns the completed generation g; the caller resumes at g+1.

    Raises
    ------
    FileNotFoundError
        If no checkpoint exists for the experiment.
    RuntimeError
        If the checkpoint's config block does not match the current run
        (the message lists every differing field), or the format version
        is unknown.
    """
    path = checkpoint_path(engine)
    if not os.path.exists(path):
        raise FileNotFoundError(f"--resume requested but no checkpoint at {path}")
    with open(path, 'rb') as f:
        state = pickle.load(f)

    if state.get('format_version') != FORMAT_VERSION:
        raise RuntimeError(
            f"Checkpoint format {state.get('format_version')} != supported {FORMAT_VERSION}")

    current = _config_block(engine)
    saved = state['config']
    # NOTE: max_generations IS part of search identity here — the quantum
    # update's cosine schedules are driven by U_total = max_generations //
    # update_quantum_gen (qnas2.py), so resuming with a different budget
    # changes the learning-rate trajectory of every remaining update. A
    # faithful resume must reuse the same max_generations.
    mismatches = [(k, saved.get(k), current.get(k))
                  for k in sorted(set(saved) | set(current))
                  if saved.get(k) != current.get(k)]
    if mismatches:
        detail = "; ".join(f"{k}: checkpoint={s!r} vs run={c!r}" for k, s, c in mismatches)
        raise RuntimeError(
            f"Checkpoint/run configuration mismatch — refusing to resume. {detail}")

    qn, qp = engine.qpop_net, engine.qpop_params
    s = state['qpop_net']
    qn.chromosome.set_num_genes(s['num_genes'])
    qn.probabilities = s['probabilities']
    qn.current_pop = s['current_pop']
    if s['current_pop_objs'] is not None:
        qn.current_pop_objs = s['current_pop_objs']
    qn.logger._update_counter = s['update_counter']
    qn.logger._last_P = s['last_P']
    qn.update_strategy._q_ema = s['q_ema']

    sp = state['qpop_params']
    qp.lower, qp.upper, qp.current_pop = sp['lower'], sp['upper'], sp['current_pop']

    engine.current_gen = state['completed_gen']
    engine.total_eval = state['total_eval']
    engine.best_so_far = state['best_so_far']
    engine.last_best_so_far = state['last_best_so_far']
    engine.best_so_far_id = state['best_so_far_id']
    engine.early_stopping_counter = state['early_stopping_counter']
    engine.random = state['random_intensity']

    if 'moqnas' in state:
        m = state['moqnas']
        engine.classical_nets = m['classical_nets']
        engine.classical_params = m['classical_params']
        engine.classical_ids = m['classical_ids']
        engine.fits = m['fits']
        engine.raw_fits = m['raw_fits']
        engine.pareto_global_population = m['pareto_global_population']
        engine.pareto_global_fitnesses = m['pareto_global_fitnesses']
        engine.pareto_global_params = m['pareto_global_params']
        engine.pareto_global_ids = m['pareto_global_ids']
        engine.fronts_history = m['fronts_history']

    # RNG restored LAST so nothing above consumes randomness afterwards.
    rng = state['rng']
    np.random.set_state(rng['numpy'])
    random.setstate(rng['python'])
    torch.set_rng_state(torch.tensor(np.asarray(rng['torch_cpu']), dtype=torch.uint8))
    if rng['torch_cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [torch.tensor(np.asarray(s), dtype=torch.uint8) for s in rng['torch_cuda']])

    engine._resumed = True
    return state['completed_gen']
