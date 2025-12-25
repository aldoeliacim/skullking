import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from app.gym_env.skullking_env_masked import SkullKingEnvMasked

# Load the latest model
model_path = "./models/masked_ppo/best_model/best_model.zip"
try:
    model = MaskablePPO.load(model_path)
except Exception:
    model = None


def mask_fn(env):
    return env.action_masks()


# Test 10 episodes
env = SkullKingEnvMasked(num_opponents=3, opponent_bot_type="random", opponent_difficulty="medium")
env = ActionMasker(env, mask_fn)

rewards = []
lengths = []
wins = 0


for _episode in range(10):
    obs, info = env.reset()
    total_reward = 0
    done = False
    steps = 0

    while not done:
        if model:
            action, _ = model.predict(obs, deterministic=True)
        else:
            mask = env.action_masks()
            valid_actions = np.where(mask == 1)[0]
            action = np.random.choice(valid_actions)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    # Check if won
    final_info = info
    if "ranking" in final_info and final_info["ranking"] == 1:
        wins += 1
        status = "🏆 WIN"
    else:
        status = f"   #{final_info.get('ranking', '?')}"

    rewards.append(total_reward)
    lengths.append(steps)
