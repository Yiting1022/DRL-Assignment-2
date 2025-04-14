import copy
import random
import math
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pickle
import math
import os
from env import Game2048Env

def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024  # 轉 MB
    print(f"[Memory Monitor] RSS: {mem:.2f} MB")


class NTupleApproximator:
    def __init__(self, board_size, patterns, num_stages=5):
        self.board_size = board_size
        self.patterns = patterns
        self.symmetry_patterns = []
        self.pattern_to_original = []
        self.weights = [[defaultdict(float) for _ in patterns] for _ in range(num_stages)]
        self.num_stages = num_stages
        self.current_stage = 0 
        
        self.stage_conditions = [
            [(4096, False), (8192, True), (16384, False)],
            [(4096, True), (8192, True), (16384, False)],
            [(2048, True), (4096, True), (8192, True), (16384, False)],
            [(16384, True)]
        ]
        
        for pattern_id, pattern in enumerate(self.patterns):
            syms = self.generate_symmetries(pattern)
            for sym in syms:
                self.symmetry_patterns.append((pattern_id, sym))
    
    def _has_tile(self, board, tile_value):
        return np.any(board == tile_value)
    
    def get_stage(self, board):
        for stage, conditions in enumerate(self.stage_conditions, 1):
            if all((self._has_tile(board, tile) if present else not self._has_tile(board, tile)) 
                for tile, present in conditions):
                return stage
        return 0
    
    def generate_symmetries(self, pattern): 
        def rot90(r,c):
            return c, self.board_size-1-r
        def flip(r,c):
            return r, self.board_size-1-c

        symmetries = []
        orig = pattern
        symmetries.append(orig)
        
        for _ in range(3): 
            orig = [rot90(r, c) for r, c in orig]
            if orig not in symmetries:
                symmetries.append(orig)
                
        for pattern in list(symmetries): 
            flipped = [flip(r, c) for r, c in pattern]
            if flipped not in symmetries:
                symmetries.append(flipped)
        
        return symmetries
    
    def tile_to_index(self, tile):
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board, coords):
        values = [board[x, y] for x, y in coords]
        indices = tuple(self.tile_to_index(tile) for tile in values)
        return indices


    def value(self, board, delta=None):
        stage = self.get_stage(board)
        weights = self.weights[stage]
        
        total_value = 0
        for pattern_id, coords in self.symmetry_patterns:
            feature = self.get_feature(board, coords)
            if delta is not None:
                weights[pattern_id][feature] += delta
            total_value += weights[pattern_id][feature]
        return total_value / len(self.symmetry_patterns)

    def update(self, board, delta, alpha):
        stage = self.get_stage(board)
        self.value(board, delta=alpha * delta)

    def best_action(self, env):

        stage = self.get_stage(env.board)
        legal_moves = [a for a in range(4) if env.is_move_legal(a)]
        if not legal_moves:
            return None

        best_value = float('-inf')
        best_actions = []

        for action in legal_moves:
            score = env.score
            temp_env = env.clone()
            next_state, reward, _, _ = temp_env.step_no_random_tile(action)

            after_state_value = self.value(next_state)
            next_value = (reward - score) + after_state_value

            if next_value > best_value:
                best_value = next_value
                best_actions = [action]
            elif next_value == best_value:
                best_actions.append(action)
        return random.choice(best_actions)
    



def mstd_learning(env, patterns, num_stages=5, episodes_per_stage=50000, alpha=0.01, save_dir="checkpoints", load_from_checkpoint=False, checkpoint_path=None):
    
    os.makedirs(save_dir, exist_ok=True)
    if load_from_checkpoint and checkpoint_path:
        print(f"Loading model from {checkpoint_path}")
        approximator = load_approximator(checkpoint_path)
        all_samples = load_samples(f"{save_dir}/stage{approximator.current_stage + 1}_samples.pkl") if os.path.exists(f"{save_dir}/stage{approximator.current_stage + 1}_samples.pkl") else [[] for _ in range(num_stages)]
    else:
        approximator = NTupleApproximator(board_size=4, patterns=patterns, num_stages=num_stages)
        all_samples = [[] for _ in range(num_stages)]
    
    try:
        next_stage_samples = train_stage(env, approximator, 0, None, episodes_per_stage, alpha)
        all_samples[1] = next_stage_samples
        save_approximator(approximator, f"{save_dir}/stage1_model.pkl")
        save_samples(next_stage_samples, f"{save_dir}/stage2_samples.pkl")
        for stage in range(1, num_stages-1):
            print(f"Stage: {stage+1}")
            next_stage_samples = train_stage(env, approximator, stage, all_samples[stage], episodes_per_stage, alpha)
            all_samples[stage+1] = next_stage_samples
            save_approximator(approximator, f"{save_dir}/stage{stage+1}_model.pkl")
            save_samples(next_stage_samples, f"{save_dir}/stage{stage+2}_samples.pkl")
        
        print(f"Stage: {num_stages}")
        train_stage(env, approximator, num_stages-1, all_samples[num_stages-1], episodes_per_stage, alpha, collect_next_stage=False)
        save_approximator(approximator, f"{save_dir}/final_model.pkl")
        return approximator, all_samples
    except KeyboardInterrupt:
        print("\nInterrupted")
        save_approximator(approximator, f"{save_dir}/current_model_stage{approximator.current_stage}.pkl")
        save_samples(all_samples[approximator.current_stage], f"{save_dir}/current_samples_stage{approximator.current_stage}.pkl")
        

def train_stage(env, approximator, stage, samples, num_episodes, alpha, collect_next_stage=True, max_samples=10000, save_dir="mstd_checkpoints"):
    

    final_scores = []
    max_tiles = []
    advance = 0

    next_stage_samples = []

    for episode in tqdm(range(num_episodes)):
        # 如果已經收集到足夠的下一階段樣本，則提前結束該階段的訓練
        if collect_next_stage and len(next_stage_samples) >= max_samples:
            print(f"Collected required {max_samples} next stage samples. Terminating stage training early.")
            break

        if stage == 0 or not samples:
            state = env.reset()
        else:
            sample = random.choice(samples)
            state = sample['state'].copy()
            env.set(state, sample['score'])

        current_stage = approximator.get_stage(state) or 0

        trajectories = []
        done = False
        max_tile = np.max(state)

        while not done:
            legal_moves = [a for a in range(4) if env.is_move_legal(a)]
            if not legal_moves:
                break

            action = approximator.best_action(env)
            if action is None:
                break

            temp_env = env.clone()
            after_step, _, _, _ = temp_env.step_no_random_tile(action)
            next_state, new_score, done, _ = env.step(action)

            # 收集符合條件的 next stage 樣本
            if collect_next_stage and len(next_stage_samples) < max_samples:
                next_stage = approximator.get_stage(next_state) or 0
                if current_stage == stage and next_stage == stage + 1:
                    advance += 1
                    next_stage_samples.append({
                        'state': next_state.copy(),
                        'score': new_score
                    })
                    if len(next_stage_samples) % 10 == 0:
                        print(f"Collected {len(next_stage_samples)}/{max_samples} Stage {stage+1} to {stage+2} Samples")

            max_tile = max(max_tile, np.max(next_state))
            current_stage = approximator.get_stage(next_state) or 0

            trajectories.append({
                "after_step": after_step.copy(),
                "next_state": next_state.copy(),
                "score": new_score
            })

        final_scores.append(env.score)
        max_tiles.append(max_tile)

        env.reset()
        for i in range(len(trajectories) - 2, -1, -1):
            transition = trajectories[i]
            score = transition["score"]
            env.set(transition["next_state"].copy(), score)
            best_action = approximator.best_action(env)

            if best_action is not None:
                next_state_after_step, score_next, _, _ = env.step_no_random_tile(best_action)
                v_next_after = approximator.value(next_state_after_step)
                v_current = approximator.value(transition["after_step"])
                delta = (score_next - score) + v_next_after - v_current
                approximator.update(transition["after_step"], delta, alpha)

        if (episode + 1) % 100 == 0:
            avg_score = np.mean(final_scores[-100:])
            max_score = np.max(final_scores[-100:])
            advance_rate = advance / (episode + 1)
            print(f"Stage {stage+1} - Episode {episode+1}/{num_episodes} | Max Score {max_score} | Average Score: {avg_score:.2f} | Advance Rate: {advance_rate:.2f} | Max Tile: {max(max_tiles[-100:])} | Avg Max Tile: {np.mean(max_tiles[-100:]):.2f}")
            print(f"Collected {len(next_stage_samples)} Next Stage Samples")
        if (episode + 1) % 2000 == 0:
            save_approximator(approximator, f"{save_dir}/checkpoint_stage{stage+1}_ep{episode+1}.pkl")
            save_samples(next_stage_samples, f"{save_dir}/checkpoint_samples_stage{stage+1}_ep{episode+1}.pkl")


    return next_stage_samples


def save_approximator(approximator, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(approximator, f)
    print(f"Saved approximator to {filepath}")


def save_samples(samples, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(samples, f)
    print(f"Saved samples to {filepath}")


def load_approximator(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def load_samples(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def import_weights_from_npz(filepath, approximator):
    data = np.load(filepath)
    for key in data.files:
        parts = key.split("_")
        stage_idx = int(parts[0][1:])
        pattern_idx = int(parts[1][1:])
        feature = tuple(map(int, parts[2:]))
        approximator.weights[stage_idx][pattern_idx][feature] = float(data[key])
    print(f"Loaded weights from {filepath}")

def export_weights_to_npz(approximator, filepath):
    data = {}
    for stage_idx, stage in enumerate(approximator.weights):
        for pattern_idx, pattern_table in enumerate(stage):
            for feature, weight in pattern_table.items():
                key = f"s{stage_idx}_p{pattern_idx}_{'_'.join(map(str, feature))}"
                data[key] = weight
    np.savez_compressed(filepath, **data)
    print(f"Saved weights to {filepath}")




if __name__ == "__main__":
    from env import Game2048Env
    patterns = [
        [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)],
        [(1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1)],
        [(2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1)],
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    ]
    
    env = Game2048Env()
    
    approximator, samples = mstd_learning(
        env=env,
        patterns=patterns,
        num_stages=5,
        episodes_per_stage=100000,
        alpha=0.01,
        save_dir="mstd_checkpoints-1",  
        load_from_checkpoint=True,
        checkpoint_path="modek88888.pkl"  # Replace with your checkpoint path
    )
