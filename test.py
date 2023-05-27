import numpy as np
from EldenEnv import EldenEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper

model_type = "PPO"
model_name = "PPO"
iteration_start = 79

models_dir = f"models/{model_name}/"
log_dir = f"logs/{model_name}/"			
model_path = f"{models_dir}/{model_name}_{iteration_start}.zip"

rng = np.random.default_rng(0)
env = EldenEnv()
expert  = PPO.load(model_path, env=env, tensorboard_log=log_dir)
# expert.learn(1000)

rollouts = rollout.rollout(
    expert,
    DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
    rollout.make_sample_until(min_timesteps=None, min_episodes=50),
    rng=rng,
)
transitions = rollout.flatten_trajectories(rollouts)

bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    rng=rng,
)
bc_trainer.train(n_epochs=1)
reward, _ = evaluate_policy(bc_trainer.policy, env, 10)
print("Reward:", reward)