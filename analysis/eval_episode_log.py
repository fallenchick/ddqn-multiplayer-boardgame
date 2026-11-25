#!/usr/bin/env python
"""Statistical analysis of a raw episode log.

Reads an ``Episode result_*.txt`` file (one line per game, each line a
``[(name, score), (name, score)]`` tuple as written by the training loop) and
reports the statistics used throughout the README:

* win rate for the DDQN agent,
* mean / standard deviation of the agent's score,
* the **coefficient of variation σ/μ** — the stability metric that replaces the
  plain variance, because variance is not comparable across opponents with
  different score scales,
* Welch's t-test on the paired score samples,
* Pearson correlation between the two players' scores.

Usage::

    python analysis/eval_episode_log.py "data/Episode result_eval-2020-Git.txt"
"""
import argparse
import ast
import os

# Keep BLAS single-threaded: the statistics here are tiny, and multi-threaded
# OpenBLAS can fail to allocate its thread buffers on constrained machines.
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np
from scipy import stats


def read_log(path):
    """Parse the log into {player_name: [scores]} preserving order."""
    per_player = {}
    order = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                games = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                continue
            for name, score in games:
                if name not in per_player:
                    per_player[name] = []
                    order.append(name)
                per_player[name].append(float(score))
    return per_player, order


def report(path, agent_name=None):
    per_player, order = read_log(path)
    if len(order) < 2:
        print('need at least two players in %s (found %s)' % (path, order))
        return None

    if agent_name is None:
        # the DDQN agent is the one that is not the fixed baseline
        agent_name = next((n for n in order if n.lower() not in ('git', 'bot1', 'ai')),
                          order[0])
    opp = next(n for n in order if n != agent_name)

    a = np.asarray(per_player[agent_name], dtype=float)
    b = np.asarray(per_player[opp], dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    wins = int((a > b).sum())
    mu, sigma = a.mean(), a.std(ddof=1)
    cv = sigma / mu if mu else float('nan')
    t_stat, p_val = stats.ttest_rel(a, b)
    r = stats.pearsonr(a, b)[0]

    print('═' * 64)
    print('  %s' % os.path.basename(path))
    print('─' * 64)
    print('  agent / opponent   : %s  vs  %s' % (agent_name, opp))
    print('  games              : %d' % n)
    print('  win rate           : %.2f %%   (%d / %d)' % (wins / n * 100, wins, n))
    print('  agent  μ / σ       : %.2f / %.2f' % (mu, sigma))
    print('  agent  σ/μ         : %.3f      <- stability (scale-free)' % cv)
    print('  opponent μ / σ     : %.2f / %.2f' % (b.mean(), b.std(ddof=1)))
    print('  mean Δ score       : %.2f' % (a - b).mean())
    print('  t-stat / p-value   : %.3f / %.3g' % (t_stat, p_val))
    print('  Pearson r          : %.3f' % r)
    print('═' * 64)
    return {
        'games': n, 'win_rate': wins / n * 100, 'mu': mu, 'sigma': sigma,
        'cv': cv, 't_stat': t_stat, 'p_value': p_val, 'pearson_r': r,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('log', help='path to an Episode result_*.txt file')
    p.add_argument('--agent', default=None,
                   help='name of the DDQN agent (auto-detected by default)')
    args = p.parse_args()
    report(args.log, args.agent)


if __name__ == '__main__':
    main()
