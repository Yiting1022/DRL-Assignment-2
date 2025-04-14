import pickle
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import gym
from gym import spaces
import copy
class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4  # 4x4 2048 board
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True  # Record if the last move was valid

        self.reset()

    def reset(self):
        """Reset the environment"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        """Compress the row: move non-zero values to the left"""
        new_row = row[row != 0]  # Remove zeros
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')  # Pad with zeros on the right
        return new_row

    def merge(self, row):
        """Merge adjacent equal numbers in the row"""
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        """Move the board left"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        """Move the board right"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            # Reverse the row, compress, merge, compress, then reverse back
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        """Move the board up"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        """Move the board down"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            # Reverse the column, compress, merge, compress, then reverse back
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        """Check if there are no legal moves left"""
        # If there is any empty cell, the game is not over
        if np.any(self.board == 0):
            return False

        # Check horizontally
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False

        # Check vertically
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True

    def step(self, action):
        """Execute one action"""
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved  # Record if the move was valid

        if moved:
            self.add_random_tile()

        done = self.is_game_over()

        return self.board, self.score, done, {}

    def step_no_random_tile(self, action):
        """Execute one action without adding a random tile"""
        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()

        done = self.is_game_over()

        return self.board, self.score, done, {}
    
    def set(self, board, score):
        """Set the board state and score"""
        self.board = board
        self.score = score
    
    def get_empty_cells(self):
        """Get list of empty cells on the board"""
        return list(zip(*np.where(self.board == 0)))

    def render(self, mode="human", action=None):
        """
        Render the current board using Matplotlib.
        This function does not check if the action is valid and only displays the current board state.
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")  # Default dark color
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        
        if mode == "human":
            plt.show()
        elif mode == "rgb_array":
            # Convert canvas to NumPy array
            fig.canvas.draw()
            image = np.array(fig.canvas.renderer.buffer_rgba())
            plt.close(fig)  # Close figure to save memory
            return image
            
    def clone(self):
        """Create a deep copy of the current environment"""
        cloned = Game2048Env()
        cloned.set(self.board.copy(), self.score)
        return cloned

    def simulate_row_move(self, row):
        """Simulate a left move for a single row"""
        # Compress: move non-zero numbers to the left
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        # Merge: merge adjacent equal numbers (do not update score)
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        # Compress again
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        """Check if the specified move is legal (i.e., changes the board)"""
        # Create a copy of the current board state
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
                # Reverse the column, simulate, then reverse back
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")

        # If the simulated board is different from the current board, the move is legal
        return not np.array_equal(self.board, temp_board)

class TreeNode:
    def __init__(self, board, score, parent=None, action=None, is_max_node=False, reward=0):
        self.board = board.copy()
        self.score = score
        self.parent = parent
        self.children = {}
        self.action = action
        self.is_max_node = is_max_node
        self.visits = 0
        self.reward = reward
    
        if self.is_max_node:
            sim_env = Game2048Env()
            sim_env.set(board.copy(), score)
            self.untried_actions = [a for a in range(4) if sim_env.is_move_legal(a)]
        else:
            self.untried_actions = []
        self.visits = 0
        self.total_reward = 0

    def fully_expanded(self):
        return len(self.children) > 0 

class MCTS:
    def __init__(self, env, approximator, iterations=500, exploration_constant=1.41, rollout_depth=0, gamma=1):
        self.env = env.clone()
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma
    
    def create_env_from_state(self, board, score):
        new_env = Game2048Env()
        new_env.set(board.copy(), score)
        return new_env

    def select_child(self, node):
        if node.is_max_node:
            best_child = None
            best_score = -float('inf')
            for child in node.children.values():
                if child.visits == 0:
                    return child
                uct_score = (
                    child.total_reward / child.visits + self.c * np.sqrt(np.log(node.visits) / child.visits)
                )
                if uct_score > best_score:
                    best_score = uct_score
                    best_child = child
            return best_child
        
        elif not node.is_max_node:  
            env = self.create_env_from_state(node.board, node.score)
            env.add_random_tile()
            if (tuple(map(tuple, env.board)), node.score) not in node.children:
                child = TreeNode(
                    board=env.board.copy(),
                    score=env.score,
                    parent=node,
                    action=None,
                    is_max_node=True
                )
                node.children[tuple(map(tuple, env.board)), node.score] = child
            return node.children[tuple(map(tuple, env.board)), node.score]
            


    def expand_chance_nodes(self, node):
        env = self.create_env_from_state(node.board, node.score)
        value = self.approximator.value(node.board)
        env.add_random_tile()   
        child  = TreeNode(
            board=env.board.copy(),
            score=env.score,
            parent=node,
            action=None,
            is_max_node=True,
        )
        node.children[tuple(map(tuple, child.board)), node.score] = child
        return value

    def expand_max_nodes(self, node):
        
        max_value = -float('inf')
        for action in node.untried_actions:
            env = self.create_env_from_state(node.board, node.score)
            next_state, reward, _, _ = env.step_no_random_tile(action)
            child = TreeNode(
                board=next_state.copy(),
                score=reward,
                parent=node,
                action=action,
                is_max_node=False,
                reward = reward - node.score   
            )
            node.children[action] = child
            value = self.approximator.value(child.board)
            if value > max_value:
                max_value = value
        node.untried_actions = []
    
        return max_value

    def backpropagate(self, node, reward):
        accumulated_reward = 0
        while node is not None:
            accumulated_reward += node.reward
            node.visits += 1
            node.total_reward += (reward + accumulated_reward)/20000
            reward *= self.gamma
            node = node.parent

    def run_simulation(self, root, score):
        node = root
        sim_env = self.create_env_from_state(node.board, node.score)

        while node.fully_expanded():
            node = self.select_child(node)
            if node is None:
                return
            sim_env.set(node.board.copy(), node.score)

        if not sim_env.is_game_over() and node.is_max_node:
            value = self.expand_max_nodes(node)
        elif not sim_env.is_game_over() and not node.is_max_node:
            value = self.expand_chance_nodes(node)
        else:
            value = 0

        self.backpropagate(node, value)
        
    def best_action_distribution(self, root):
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        
        return best_action, distribution

def run(env, approximator):
    state = env.reset()
    td_mcts = MCTS(env, approximator, iterations=100, exploration_constant=1.41, rollout_depth=0, gamma=1)
    done = False
    root = TreeNode(state, env.score, is_max_node=True)
    while not done:
        for _ in range(td_mcts.iterations):
            td_mcts.run_simulation(root, env.score)

        best_act, action_distribution = td_mcts.best_action_distribution(root)
        state, reward, done, _ = env.step(best_act)
        root = root.children[best_act]  # Move to the selected 
        root.parent = None
    
        if (tuple(map(tuple, state)), env.score) not in root.children:
            root.children[(tuple(map(tuple, state)), env.score)] = TreeNode(env.board.copy(), env.score, parent=root, is_max_node=True)
        root = root.children[(tuple(map(tuple, state)), env.score)]  # Move to the selected child node
        #frames.append(env.render(action=best_act, mode="rgb_array"))
        #break

    print("Game over, final score:", env.score)
    return env.score
    

