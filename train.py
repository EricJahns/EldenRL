from stable_baselines3 import PPO, A2C
from sb3_contrib import RecurrentPPO, QRDQN
from stable_baselines3 import HerReplayBuffer
import os
from EldenEnv import EldenEnv

RESUME = True
TIMESTEPS = 1
HORIZON_WINDOW = 1000

iteration_start = 7

# model_type = "QRDQN"
model_type = "RecurrentPPO"
# model_type = "large_PPO"
# model_type = "large_PPO"
# model_ty = "A2C"

model_name = "RecurrentPPO_Soldier_Godrick"

def setup_files() -> tuple:
    if not os.path.exists(f"models/{model_name}/"):
        os.makedirs(f"models/{model_name}/")
    if not os.path.exists(f"logs/{model_name}/"):
        os.makedirs(f"logs/{model_name}/")

    models_dir = f"models/{model_name}/"
    log_dir = f"logs/{model_name}/"			
    model_path = f"{models_dir}/{model_name}_{iteration_start}.zip"

    print("Model: Folder structure created...")

    return models_dir, log_dir, model_path

def setup_enviroment() -> EldenEnv:
    print("EldenEnv initialized...")
    return EldenEnv()

def get_ppo(env: EldenEnv, log_dir: str) -> PPO:
    return PPO('MultiInputPolicy',
                env,
                tensorboard_log=log_dir,
                n_steps=HORIZON_WINDOW,
                verbose=2,
                batch_size=20,
                normalize_advantage=True,
                clip_range=0.2,
                clip_range_vf=0.2,
                gae_lambda=0.9,
                learning_rate=3e-4,
                device='cuda')

def get_large_ppo(env: EldenEnv, log_dir: str) -> PPO:
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128, 64], vf=[128, 128, 64])
    )

    return PPO('MultiInputPolicy',
                env,
                tensorboard_log=log_dir,
                n_steps=HORIZON_WINDOW,
                policy_kwargs=policy_kwargs,
                verbose=2,
                gae_lambda=0.9,
                batch_size=20,
                normalize_advantage=True,
                clip_range=0.4,
                clip_range_vf=0.2,
                learning_rate=3e-4,
                device='cuda')

def get_recurrent_ppo(env: EldenEnv, log_dir: str) -> RecurrentPPO:
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128, 64], vf=[128, 128, 64])
    )

    return RecurrentPPO('MultiInputLstmPolicy',
                        env,
                        tensorboard_log=log_dir,
                        n_steps=HORIZON_WINDOW,
                        policy_kwargs=policy_kwargs,
                        verbose=2,
                        batch_size=20,
                        normalize_advantage=True,
                        clip_range=0.5,
                        clip_range_vf=0.5,
                        gae_lambda=0.9,
                        learning_rate=9e-4,
                        device='cuda')

def get_qrdqn(env: EldenEnv, log_dir: str) -> QRDQN:
    return QRDQN('MultiInputPolicy',
                  env,
                  tensorboard_log=log_dir,
                  train_freq=500,
                  buffer_size=10000,
                  learning_starts=1000,
                  batch_size=32,
                  target_update_interval=2000,
                  replay_buffer_class=HerReplayBuffer,
                  device='cuda')

def get_a2c(env: EldenEnv, log_dir: str) -> A2C:
    return A2C('MultiInputPolicy',
                env,
                tensorboard_log=log_dir,
                n_steps=HORIZON_WINDOW,
                verbose=2,
                normalize_advantage=True,
                use_rms_prop=True,
                gae_lambda=0.9,
                learning_rate=3e-4,
                device='cuda')

def setup_model(model_path: str, log_dir: str) -> None:
    init_models_dict = {
        "QRDQN": get_qrdqn,
        "RecurrentPPO": get_recurrent_ppo,
        "PPO": get_ppo,
        "A2C": get_a2c,
        "large_PPO": get_large_ppo
    }

    resume_models_dict = {
        "QRDQN": QRDQN,
        "RecurrentPPO": RecurrentPPO,
        "PPO": PPO,
        "A2C": A2C,
        "large_PPO": PPO
    }

    print("EldenRL - Train.py")
    print(f"Model: Setting up a {model_name} model...")

    env = setup_enviroment()
    model = None
    if not RESUME:
        model = init_models_dict[model_type](env, log_dir)
        print("Model: Model initialized...")
    else:
        model = resume_models_dict[model_type].load(model_path, env=env, tensorboard_log=log_dir)
        print("Model: Model loaded...")

    return model
    
def train() -> None:     
    models_dir, logdir, model_path = setup_files()
    model = setup_model(model_path, logdir)

    iters = iteration_start
    while True:	
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, log_interval=1, reset_num_timesteps=False)
        if iters == 1 or iters == 10 or iters == 50 or iters == 100 or iters == 150:
            model.save(f"{models_dir}/{model_name}_{iters}")
        else:
            model.save(f"{models_dir}/{model_name}_{iters}")
        print(f"Model: Model saved {iters} times.")

if __name__ == "__main__":
    train()