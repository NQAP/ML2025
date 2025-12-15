import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FuncAnimation
import os
from PIL import Image
from tqdm import tqdm

# Define constants
BOARD_SIZE = 8
MAX_ROUNDS = 50
OBSTACLES = {(3, 3), (3, 4), (4, 3), (4, 4), (2,2), (2,5), (5,2), (5,5)}
PAWN_MOVE_PROB = 0.2

# ==========================================
# Modification 1 & 2 for Faster Play
# ==========================================
# 原本 REWARD_STEP = 0，Agent 容易在原地打轉或走遠路。
# 修改 1: 給予每一步負回饋 (REWARD_STEP = -0.1)，強迫 Agent 尋找最短路徑。
# 修改 2: 提高捕捉獎勵 (REWARD_CATCH = 5.0)，增大目標與過程的 Value 差異 (Gradient)。
# 修改 3: 稍微降低 Gamma (0.95)，讓 Agent 更看重近期的獎勵 (雖非必要，但有助於收斂速度)。

REWARD_CATCH = 5.0      # Increased reward
REWARD_STEP = -0.1      # Time penalty
GAMMA = 0.95            # Discount factor
TAU = 0.1               # Learning rate
num_episodes = 100000

class ChessEnvironment:
    def __init__(self):
        self.reset()

    def reset(self):
        self.knight_pos = self._random_position()
        self.pawn_pos = self._random_position()
        self.rounds = 0
        return self._get_state()

    def _random_position(self):
        while True:
            pos = (random.randint(0, BOARD_SIZE - 1), random.randint(0, BOARD_SIZE - 1))
            if pos not in OBSTACLES:
                return pos

    def _get_state(self):
        return (*self.knight_pos, *self.pawn_pos)

    def _knight_moves(self, pos):
        x, y = pos
        moves = [
            (x + 2, y + 1), (x + 2, y - 1), (x - 2, y + 1), (x - 2, y - 1),
            (x + 1, y + 2), (x + 1, y - 2), (x - 1, y + 2), (x - 1, y - 2)
        ]
        return [
            (nx, ny) for nx, ny in moves
            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and (nx, ny) not in OBSTACLES
        ]

    def step(self, knight_action):
        # Update knight position
        self.knight_pos = knight_action

        # Check for termination
        if self.knight_pos == self.pawn_pos:
            return self._get_state(), REWARD_CATCH, True

        # Pawn's random movement
        if random.random() < PAWN_MOVE_PROB:
            px, py = self.pawn_pos
            if px < BOARD_SIZE - 1 and (px + 1, py) not in OBSTACLES:
                self.pawn_pos = (px + 1, py)

        self.rounds += 1
        done = self.rounds >= MAX_ROUNDS
        return (self._get_state()), REWARD_STEP, done

# TODO : Action selection strategies
def greedy_action_selection(env, state, value_table):
    # Find the action with the highest value
    kx, ky, px, py = state
    actions = env._knight_moves((kx, ky)) # -> List[(knight_x, knight_y)]
    
    if not actions:
        return (kx, ky) # Stay if no moves (should not happen in valid grid)

    best_action = None
    max_val = -float('inf')

    # Look ahead to see which neighbor has the highest Value
    # Note: We assume Pawn stays still for the purpose of selection (Greedy wrt current state)
    for action in actions:
        # action is the next (kx, ky)
        val = value_table[action[0], action[1], px, py]
        if val > max_val:
            max_val = val
            best_action = action
            
    return best_action

def epsilon_greedy_action_selection(env, state, value_table, epsilon=0.1):
    # TODO : With probability epsilon, take a random action
    if random.random() < epsilon:
        kx, ky = state[:2]
        actions = env._knight_moves((kx, ky))
        if not actions: return (kx, ky)
        return random.choice(actions)
    else:
        return greedy_action_selection(env, state, value_table)

def ucb_action_selection(env, state, value_table, count_table, c=2):
    # Optional implementation, we will use Epsilon-Greedy
    pass


# Tabular RL
def train_agent():
    env = ChessEnvironment()
    # state = (knight_x, knight_y, pawn_x, pawn_y)
    value_table = np.zeros((BOARD_SIZE, BOARD_SIZE, BOARD_SIZE, BOARD_SIZE)) 

    # Epsilon decay strategy
    epsilon = 0.9
    epsilon_min = 0.01
    epsilon_decay = 0.99995

    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        done = False

        while not done:
            # Select Action (Using Epsilon-Greedy)
            action = epsilon_greedy_action_selection(env, state, value_table, epsilon)

            # Take the action
            next_state, reward, done = env.step(action)

            # Compute source value (Target for TD Learning)
            # We want V(S) -> Reward + Gamma * V(S')
            # Since S' is determined by Environment (including Pawn move), we look at next_state
            
            # Note: The template code structure suggested calculating max_next_value from next_actions
            # This is essentially checking: What is the best value attainable from the new state?
            nkx, nky, npx, npy = next_state
            
            if done:
                target = reward
            else:
                # Standard TD(0): V(S')
                # target = reward + GAMMA * value_table[nkx, nky, npx, npy]
                
                # Alternative (consistent with template structure): 
                # Bellman Optimality looking ahead one step from next_state 
                # (effectively V(S') approx max Q(S', a'))
                next_actions = env._knight_moves((nkx, nky))
                if next_actions:
                    max_next_value = max([value_table[nx, ny, npx, npy] for nx, ny in next_actions])
                else:
                    max_next_value = value_table[nkx, nky, npx, npy] # Fallback
                
                target = reward + GAMMA * max_next_value

            # Update the value table (Soft update / TD Learning)
            # V(S) = V(S) + alpha * (Target - V(S))
            current_val = value_table[state]
            value_table[state] = current_val + TAU * (target - current_val)

            state = next_state
        
        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

    return value_table

# Helper functions (UNCHANGED)
def finish_game(value_table, env):
    # Generate frames of a complete game with progress bar.
    state = env._get_state()
    frames = []
    done = False
    step = 0
    while not done:
        frames.append((*state, step))
        action = greedy_action_selection(env, state, value_table)
        next_state, _, done = env.step(action)
        state = next_state
        step += 1
    frames.append((*state, step))
    return frames

def render_frame(kx, ky, px, py, step, max_steps):
    fig, ax = plt.subplots(figsize=(6, 7))  # Add extra space for the progress bar

    # Plot the chessboard
    ax.set_xlim(0, BOARD_SIZE-1)
    ax.set_ylim(0, BOARD_SIZE-1)
    ax.set_xticks(range(BOARD_SIZE))
    ax.set_yticks(range(BOARD_SIZE))
    ax.grid(True)

    # Draw the obstacles at the center of grid cells
    for x, y in OBSTACLES:
        ax.text(y, BOARD_SIZE - 1 - x, "O", ha="center", va="center", fontsize=20, color="black")

    # Draw the knight at the center of its grid cell
    ax.text(ky, BOARD_SIZE - 1 - kx, "K", ha="center", va="center", fontsize=20, color="blue")

    # Draw the pawn at the center of its grid cell
    ax.text(py, BOARD_SIZE - 1 - px, "P", ha="center", va="center", fontsize=20, color="red")

    # Add a title
    ax.set_title("Catch the Pawn")

    # Add a progress bar below the chessboard
    bar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.05])  # [left, bottom, width, height]
    bar_ax.barh(0, width=step / (max_steps-1 + 0.001), height=1, color="green")
    bar_ax.set_xlim(0, 1)
    bar_ax.axis("off")
    bar_ax.text(0.5, 0, f"Step: {step}/{max_steps-1}", ha="center", va="center", fontsize=12, color="white")

    # Convert the figure to a PIL image
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
    pil_img = Image.fromarray(img)

    plt.close(fig)
    return pil_img

def save_gif(frames, filename="game.gif"):
    if not frames: return
    frames[0].save(
        filename, save_all=True, append_images=frames[1:]+frames[-1:]*3, duration=500, loop=0
    )

# Train the agent
print("Training Agent...")
value_table = train_agent()
print("Training Complete.")

"""### Evaluate on the Public cases"""

# 100 public states
public_states = [(1, 7, 7, 5), (6, 7, 7, 2), (6, 2, 6, 2), (1, 1, 1, 5), (2, 1, 1, 1), (4, 2, 1, 6), (5, 7, 5, 3), (5, 4, 1, 0), (6, 5, 2, 3), (2, 4, 1, 0), (3, 6, 5, 7), (0, 3, 1, 3), (0, 7, 5, 3), (3, 5, 1, 5), (2, 1, 7, 5), (1, 5, 1, 6), (1, 2, 6, 6), (7, 5, 4, 2), (6, 7, 7, 4), (2, 6, 7, 6), (7, 3, 7, 5), (0, 6, 7, 6), (6, 5, 1, 4), (2, 6, 2, 1), (3, 2, 6, 4), (0, 3, 2, 4), (7, 2, 5, 1), (3, 6, 1, 0), (6, 4, 0, 3), (6, 1, 5, 3), (5, 4, 4, 1), (7, 7, 1, 6), (4, 7, 5, 4), (5, 6, 0, 4), (2, 6, 0, 3), (7, 0, 1, 4), (6, 4, 1, 4), (0, 2, 3, 0), (4, 6, 1, 0), (1, 1, 7, 4), (1, 4, 6, 2), (1, 2, 4, 1), (4, 7, 0, 1), (5, 4, 2, 6), (6, 4, 6, 0), (2, 1, 1, 5), (5, 3, 2, 6), (6, 7, 2, 0), (6, 3, 0, 6), (6, 1, 3, 7), (5, 7, 1, 3), (0, 6, 1, 6), (0, 6, 6, 2), (4, 1, 1, 5), (7, 1, 3, 2), (7, 6, 1, 3), (1, 7, 1, 4), (5, 6, 1, 1), (5, 1, 6, 2), (0, 4, 2, 1), (0, 2, 1, 4), (6, 1, 7, 4), (7, 3, 3, 5), (3, 5, 6, 7), (0, 4, 6, 1), (2, 1, 7, 2), (0, 1, 5, 3), (4, 7, 3, 1), (7, 7, 4, 0), (4, 7, 2, 3), (1, 4, 1, 1), (1, 2, 1, 0), (6, 4, 0, 0), (7, 3, 1, 1), (2, 4, 1, 7), (2, 0, 4, 5), (7, 1, 5, 4), (1, 5, 1, 5), (1, 7, 2, 1), (7, 4, 3, 7), (6, 4, 2, 0), (4, 2, 6, 1), (3, 0, 0, 6), (4, 2, 0, 6), (2, 6, 6, 7), (2, 6, 7, 5), (2, 3, 3, 2), (7, 1, 3, 6), (2, 1, 6, 5), (2, 7, 7, 5), (7, 4, 7, 4), (4, 6, 6, 5), (2, 1, 2, 1), (2, 1, 5, 0), (1, 0, 0, 1), (5, 0, 3, 5), (0, 0, 3, 5), (6, 3, 0, 3), (4, 5, 6, 4), (1, 7, 3, 1)]

# Eval environment (score = 100 - rounds)
class EvalEnvironment:
    def __init__(self):
        self.reset()

    def reset(self):
        self.knight_pos = self._random_position()
        self.pawn_pos = self._random_position()
        self.rounds = 0
        return self._get_state()

    def set_state(self, state):
        """Set the state of the game."""
        self.knight_pos, self.pawn_pos = state[:2], state[2:]
        self.rounds = 0
        return self._get_state()

    def _random_position(self):
        while True:
            pos = (random.randint(0, BOARD_SIZE - 1), random.randint(0, BOARD_SIZE - 1))
            if pos not in OBSTACLES:
                return pos

    def _get_state(self):
        return (*self.knight_pos, *self.pawn_pos)

    def _knight_moves(self, pos):
        x, y = pos
        moves = [
            (x + 2, y + 1), (x + 2, y - 1), (x - 2, y + 1), (x - 2, y - 1),
            (x + 1, y + 2), (x + 1, y - 2), (x - 1, y + 2), (x - 1, y - 2)
        ]
        return [
            (nx, ny) for nx, ny in moves
            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and (nx, ny) not in OBSTACLES
        ]

    def step(self, knight_action):
        # Update knight position
        self.knight_pos = knight_action

        # Check for termination
        if self.knight_pos == self.pawn_pos:
            return self._get_state(), 100-self.rounds, True

        # Pawn's random movement
        if random.random() < PAWN_MOVE_PROB:
            px, py = self.pawn_pos
            if px < BOARD_SIZE - 1 and (px + 1, py) not in OBSTACLES:
                self.pawn_pos = (px + 1, py)

        self.rounds += 1
        done = self.rounds >= MAX_ROUNDS
        return (self._get_state()), 0, done

def finish_game_eval(value_table, env):
    # Finish the game and get the score
    state = env._get_state()
    done = False
    while not done:
        action = greedy_action_selection(env, state, value_table)
        state, score, done = env.step(action)
    return score

# Evaluate the value table
print("Evaluating...")
env = EvalEnvironment()

# fix the random seed (TA will use this seed)
random.seed(42)

# run the public cases
scores = []
for state in public_states:
    env.set_state(state)
    score = finish_game_eval(value_table, env)
    scores.append(score)
print(f"Public score: {sum(scores)/len(scores)}")

# Save for submission
np.save("value_table.npy", value_table)