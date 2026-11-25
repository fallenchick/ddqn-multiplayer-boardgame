#!/usr/bin/env python
"""Training entry point for the DDQN agent on *Railroad Tour*.

Examples
--------
Train from scratch on the USA map, 2 players (the configuration reported in
the README)::

    python train.py --episodes 2050 --learn

Resume from a checkpoint::

    python train.py --episodes 2050 --learn --resume checkpoints/ddqn_checkpoint_episode_1000.pth

Evaluate only (no learning)::

    python train.py --episodes 100 --evaluate-only \
        --checkpoint checkpoints/ddqn_checkpoint_episode_2020.pth --opponent github
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

from ddqn import config, env
from ddqn.baselines import AIPlayer, GithubPlayer
from ddqn.ddqn_agent import DDQNPlayer


def build_game(gametype=config.GAMETYPE, player_num=config.PLAYER_NUM):
    """Create a fresh game and attach the agents.

    The DDQN agent is always the second seat; the opponent is chosen by
    ``--opponent``.  The shared instance is published on ``env.game`` *before*
    anything touches it, because the agent sizes its network from the route
    table and the distance-matrix helper reads the city count from the game.
    """
    game = env.Game()
    game.specialsetup(gametype)
    env.game = game                              # publish first
    game.dist_matrix = env.create_distance_matrix(game.route_list)

    opponent = {
        'heuristic': lambda: AIPlayer('Bot1'),
        'github': lambda: GithubPlayer('Git'),
    }[args.opponent]()

    agent = DDQNPlayer('DDQN')
    game.players = [opponent, agent]
    random.shuffle(game.players)
    env.game = game                              # re-publish (same object)
    return game, agent


def run_episode(game, agent, eps):
    """Play one full episode and return (agent_score, opponent_score)."""
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
            print('Invalid eps — episode aborted')
            break

    scores = [(p.name, p.calc_end_score()) for p in game.players]
    return scores


def save_checkpoint(agent, episode, path):
    torch.save({
        'episode': episode,
        'epsilon': agent.e_greed,
        'online_net_state_dict': agent.online_net.state_dict(),
        'target_net_state_dict': agent.target_net.state_dict(),
        'optimizer_state_dict': agent.opt.state_dict(),
    }, path)


def load_checkpoint(agent, path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    agent.e_greed = ck['epsilon']
    agent.online_net.load_state_dict(ck['online_net_state_dict'])
    agent.target_net.load_state_dict(ck['target_net_state_dict'])
    agent.opt.load_state_dict(ck['optimizer_state_dict'])
    return ck.get('episode', 0)


def main():
    global args
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--episodes', type=int, default=config.TRAIN_EPISODES,
                   help='number of episodes to run')
    p.add_argument('--learn', action='store_true',
                   help='enable learning (writes to the training log)')
    p.add_argument('--evaluate-only', action='store_true',
                   help='disable learning; evaluation is the default')
    p.add_argument('--opponent', choices=['heuristic', 'github'], default='heuristic',
                   help='which baseline to face')
    p.add_argument('--checkpoint', type=str, default=None,
                   help='checkpoint to load before starting')
    p.add_argument('--resume', type=str, default=None,
                   help='checkpoint to resume training from')
    p.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                   help='where to write checkpoints')
    p.add_argument('--log', type=str, default='data/Episode result_learn.txt',
                   help='episode log to append to')
    p.add_argument('--seed', type=int, default=None, help='RNG seed')
    p.add_argument('--gametype', type=int, default=config.GAMETYPE,
                   help='1 = USA map, 2 = Europe map')
    p.add_argument('--verbose', action='store_true',
                   help='print per-turn debug output (slow; off by default)')
    args = p.parse_args()

    config.VERBOSE = args.verbose
    config.LEARN = bool(args.learn)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log) or '.', exist_ok=True)

    # The DDQN agent needs the game object at construction time (it sizes its
    # input/output layers from the route table), so build the game first.
    game, agent = build_game()
    env.game = game                       # publish the shared instance

    start_episode = 0
    if args.resume:
        start_episode = load_checkpoint(agent, args.resume)
        print('[*] resumed from %s (episode %d, eps=%.4f)'
              % (args.resume, start_episode, agent.e_greed))
    elif args.checkpoint:
        load_checkpoint(agent, args.checkpoint)
        print('[*] loaded %s' % args.checkpoint)

    log_path = args.log if args.learn else 'data/Episode result_eval.txt'
    mode = 'a' if args.learn else 'w'
    print('[*] %s %d episodes | opponent=%s | %s'
          % ('training' if args.learn else 'evaluating',
             args.episodes, args.opponent, 'resume@%d' % start_episode
             if start_episode else 'scratch'))

    with open(log_path, mode) as log:
        for ep in range(start_episode, start_episode + args.episodes):
            # a fresh game per episode, but keep the learned weights
            game, agent = build_game()
            env.game = game
            if args.resume and ep == start_episode:
                pass  # weights already loaded into `agent`; rebuild keeps them
            scores = run_episode(game, agent, ep)

            if args.learn and (ep + 1) % config.CHECKPOINT_EVERY == 0:
                save_checkpoint(agent, ep + 1,
                                os.path.join(args.checkpoint_dir,
                                             'ddqn_checkpoint_episode_%d.pth' % (ep + 1)))
            log.write(str(scores) + '\n')
            if (ep + 1) % 50 == 0:
                log.flush()
                print('  episode %d | eps=%.4f | %s'
                      % (ep + 1, agent.e_greed, scores))

    log.flush()
    print('[✓] finished — log written to %s' % log_path)


if __name__ == '__main__':
    main()
