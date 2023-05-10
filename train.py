from stable_baselines3 import PPO, A2C
from sb3_contrib import RecurrentPPO, QRDQN
from stable_baselines3 import HerReplayBuffer
import os
from EldenEnv import EldenEnv

RESUME = True
TIMESTEPS = 1
HORIZON_WINDOW = 1000

# model_name = "QRDQN"
# model_name = "RecurrentPPO"
model_name = "PPO"

def setup_files() -> tuple:
    if not os.path.exists(f"models/{model_name}/"):
        os.makedirs(f"models/{model_name}/")
    if not os.path.exists(f"logs/{model_name}/"):
        os.makedirs(f"logs/{model_name}/")

    models_dir = f"models/{model_name}/"
    log_dir = f"logs/{model_name}/"			
    model_path = f"{models_dir}/{model_name}.zip"

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
                target_kl=0.02,
                clip_range=0.2,
                clip_range_vf=0.2,
                gae_lambda=0.9,
                learning_rate=3e-4,
                device='cuda')

def get_recurrent_ppo(env: EldenEnv, log_dir: str) -> RecurrentPPO:
    return RecurrentPPO('MultiInputLstmPolicy',
                        env,
                        tensorboard_log=log_dir,
                        n_steps=HORIZON_WINDOW,
                        verbose=2,
                        batch_size=20,
                        normalize_advantage=True,
                        target_kl=0.02,
                        clip_range=0.2,
                        clip_range_vf=0.2,
                        gae_lambda=0.9,
                        learning_rate=3e-4,
                        device='cuda')

def get_qrdqn(env: EldenEnv, log_dir: str) -> QRDQN:
    return QRDQN('MultiInputPolicy',
                  env,
                  train_freq=500,
                  buffer_size=10000,
                  learning_starts=1000,
                  batch_size=32,
                  target_update_interval=2000,
                  replay_buffer_class=HerReplayBuffer,
                  device='cuda')

def setup_model(model_path: str, log_dir: str) -> None:
    init_models_dict = {
        "QRDQN": get_qrdqn,
        "RecurrentPPO": get_recurrent_ppo,
        "PPO": get_ppo
    }

    resume_models_dict = {
        "QRDQN": QRDQN,
        "RecurrentPPO": RecurrentPPO,
        "PPO": PPO
    }

    print("EldenRL - Train.py")
    print(f"Model: Setting up a {model_name} model...")

    env = setup_enviroment()

    if not RESUME:
        model = init_models_dict[model_name](env, log_dir)
        print("Model: Model initialized...")
    else:
        model = resume_models_dict[model_name].load(model_path, env=env, tensorboard_log=log_dir)
        print("Model: Model loaded...")

    return model
    
def train() -> None:     
    models_dir, logdir, model_path = setup_files()
    model = setup_model(model_path, logdir)

    iters = 48
    while True:	
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, log_interval=1, reset_num_timesteps=False)
        if iters == 1 or iters == 10 or iters == 50 or iters == 100 or iters == 150:
            model.save(f"{models_dir}/{model_name}_{iters}")
        model.save(f"{models_dir}/{model_name}_{iters}")
        print(f"Model: Model saved {iters} times.")

if __name__ == "__main__":
    train()