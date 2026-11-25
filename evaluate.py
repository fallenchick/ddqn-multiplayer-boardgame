#!/usr/bin/env python
"""Evaluate a trained checkpoint against a baseline opponent.

The evaluation protocol is fixed and identical for every checkpoint compared:

* same map, same player count, same number of games;
* **both** the win rate (central tendency) and the coefficient of variation
  σ/μ (stability) are reported — either alone is misleading, because a high
  win rate can be produced by a high-variance agent that wins big and loses
  big;
* Welch's t-test on the per-game score samples.

Examples
--------
::

    python evaluate.py --checkpoint checkpoints/ddqn_checkpoint_episode_2020.pth \
                       --opponent heuristic --games 150

    # compare two checkpoints under one protocol
    python evaluate.py --checkpoint checkpoints/ddqn_checkpoint_episode_1540.pth \
                       --opponent github --games 150
"""
import argparse
import os

# Keep BLAS single-threaded: the numerics here are small, and multi-threaded
# OpenBLAS can fail to allocate its thread buffers on constrained machines.
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import random
import sys

import numpy as np
import torch
from scipy import stats

from ddqn import config, env
from ddqn.baselines import AIPlayer, GithubPlayer
from ddqn.ddqn_agent import DDQNPlayer


def build_game(opponent_kind, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    game = env.Game()
    game.specialsetup(config.GAMETYPE)
    env.game = game
    game.dist_matrix = env.create_distance_matrix(game.route_list)

    opponent = {
        'heuristic': lambda: AIPlayer('Bot1'),
        'github': lambda: GithubPlayer('Git'),
    }[opponent_kind]()

    agent = DDQNPlayer('DDQN')
    game.players = [opponent, agent]
    random.shuffle(game.players)
    env.game = game
    return game, agent


def play_one(game, agent, eps=0.0):
    """Play a single game; return (agent_score, opponent_score)."""
    agent.DDQNinit(eps)
    for player in game.players:
        player.draw_ticket(3, 2)

    while game.turn <= game.end_turn:
        for player in game.players:
            player.dist_matrix = env.create_distance_matrix(player.routes)
            game.turn += 1
            if game.turn > game.end_turn:
                continue
            player.Turn()
        if game.turn > 200:
            break

    scores = {p.name: p.calc_end_score() for p in game.players}
    agent_score = scores.get('DDQN')
    opp_score = next(v for k, v in scores.items() if k != 'DDQN')
    return agent_score, opp_score


def summarise(agent_scores, opp_scores, label):
    a = np.asarray(agent_scores, dtype=float)
    b = np.asarray(opp_scores, dtype=float)
    wins = int((a > b).sum())
    n = len(a)
    win_rate = wins / n
    mu, sigma = a.mean(), a.std(ddof=1)
    cv = sigma / mu if mu else float('nan')
    t_stat, p_val = stats.ttest_rel(a, b) if n > 1 else (float('nan'), float('nan'))
    try:
        r = stats.pearsonr(a, b)[0]
    except Exception:
        r = float('nan')

    out = {
        'label': label,
        'games': n,
        'wins': wins,
        'win_rate': round(win_rate * 100, 1),
        'agent_mu': round(float(mu), 2),
        'agent_sigma': round(float(sigma), 2),
        'agent_cv_sigma_over_mu': round(float(cv), 3),
        'opponent_mu': round(float(b.mean()), 2),
        'opponent_sigma': round(float(b.std(ddof=1)), 2),
        'mean_score_diff': round(float((a - b).mean()), 2),
        't_stat': round(float(t_stat), 3),
        'p_value': float('%.3g' % p_val),
        'pearson_r': round(float(r), 3),
    }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', type=str, default=None,
                   help='checkpoint to evaluate (omit for an untrained agent)')
    p.add_argument('--opponent', choices=['heuristic', 'github'], default='heuristic')
    p.add_argument('--games', type=int, default=config.EVAL_GAMES,
                   help='games per arm — keep this identical across checkpoints')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=str, default=None, help='write the summary as JSON')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    config.VERBOSE = args.verbose
    config.LEARN = False            # never learn while evaluating

    game, agent = build_game(args.opponent, seed=args.seed)

    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        agent.online_net.load_state_dict(ck['online_net_state_dict'])
        agent.target_net.load_state_dict(ck['target_net_state_dict'])
        agent.e_greed = ck.get('epsilon', 0.0)
        print('[*] loaded %s (episode %s)' % (args.checkpoint, ck.get('episode', '?')))

    agent_scores, opp_scores = [], []
    for g in range(args.games):
        # a fresh board each game, same weights
        game, agent2 = build_game(args.opponent, seed=args.seed + g + 1)
        if args.checkpoint:
            agent2.online_net.load_state_dict(agent.online_net.state_dict())
            agent2.target_net.load_state_dict(agent.target_net.state_dict())
            agent2.e_greed = agent.e_greed
        a, b = play_one(game, agent2)
        agent_scores.append(a)
        opp_scores.append(b)
        if (g + 1) % 25 == 0:
            print('  %d/%d games' % (g + 1, args.games))

    label = '%s vs %s' % (args.checkpoint or 'untrained', args.opponent)
    res = summarise(agent_scores, opp_scores, label)

    print()
    print('═' * 62)
    print('  %s' % res['label'])
    print('─' * 62)
    print('  games          : %d' % res['games'])
    print('  win rate       : %.1f %%   (%d / %d)' % (res['win_rate'], res['wins'], res['games']))
    print('  score μ / σ    : %.2f / %.2f' % (res['agent_mu'], res['agent_sigma']))
    print('  σ/μ (stability): %.3f' % res['agent_cv_sigma_over_mu'])
    print('  opponent μ/σ   : %.2f / %.2f' % (res['opponent_mu'], res['opponent_sigma']))
    print('  mean Δ score   : %.2f' % res['mean_score_diff'])
    print('  t-stat / p     : %.3f / %.3g' % (res['t_stat'], res['p_value']))
    print('  Pearson r      : %.3f' % res['pearson_r'])
    print('═' * 62)

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print('[*] summary written to %s' % args.out)


if __name__ == '__main__':
    main()
