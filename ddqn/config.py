"""All hyper-parameters in one place.

These are the values locked in for the runs reported in the README
(USA map, 2 players).  Change them here rather than in the source.
"""

# ---- problem shape -----------------------------------------------------------
STATE_DIM_BASE = 142        # fixed part of the observation
STATE_DIM_PER_PLAYER = 2    # +2 per player in the game
N_ROUTES = 100              # 100 routes + 1 "wait" action
ACTION_DIM = N_ROUTES + 1

# ---- Double DQN --------------------------------------------------------------
GAMMA = 1.0                 # full horizon; ticket payoff is booked at the end
LR = 1e-3
BATCH_SIZE = 32
REPLAY_CAPACITY = 1000      # deque(maxlen=1000)
EPS_START = 1.0
EPS_MIN = 0.01
EPS_DECAY = 0.998
EPS_WARMUP = 10             # episodes of (near-)random play before learning
MEMORY_WARMUP = 128         # transitions collected before the first update

# ---- network -----------------------------------------------------------------
HIDDEN_UNITS = 256

# ---- reward ------------------------------------------------------------------
TIME_PENALTY = -0.01        # per turn, encourages fast termination

# ---- training / evaluation ---------------------------------------------------
TRAIN_EPISODES = 2050
EVAL_GAMES = 150            # games per arm in every evaluation
CHECKPOINT_EVERY = 10       # episodes between checkpoints

# ---- game --------------------------------------------------------------------
LEARN = True                # learn (True) or evaluate only (False)
VERBOSE = False             # per-turn debug output (slow); set via train.py --verbose

GAMETYPE = 1                # 1 = USA map, 2 = Europe map
PLAYER_NUM = 2
