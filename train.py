from stable_baselines3 import PPO, A2C
from sb3_contrib import RecurrentPPO, QRDQN
from stable_baselines3 import HerReplayBuffer
import os
from EldenEnv import EldenEnv

print("EldenRL - Train.py")
print("Model: Hello. Training will start soon. This can take a while to initialize...")

RESUME = True
TIMESTEPS = 1
HORIZON_WINDOW = 1000

# model_name = "QRDQN"
# model_name = "RecurrentPPO"
model_name = "PPO"

if not os.path.exists(f"models/{model_name}/"):
	os.makedirs(f"models/{model_name}/")
if not os.path.exists(f"logs/{model_name}/"):
	os.makedirs(f"logs/{model_name}/")
models_dir = f"models/{model_name}/"
logdir = f"logs/{model_name}/"			
model_path = f"{models_dir}/{model_name}.zip"
print("Model: Folder structure created...")


env = EldenEnv()
# env = CustomEnvWrapper([lambda: env])

print("EldenEnv initialized...")

if not RESUME:	
	# model = RecurrentPPO('MultiInputLstmPolicy',
    model = PPO('MultiInputPolicy',
                env,
                tensorboard_log=logdir,
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
    
    # model = QRDQN('MultiInputPolicy',
    #               env,
    #               train_freq=500,
    #               buffer_size=10000,
    #               learning_starts=1000,
    #               batch_size=32,
    #               target_update_interval=2000,
    #               replay_buffer_class=HerReplayBuffer,
    #               device='cuda')
    print("Model: Model initialized...")

else:
    # model = RecurrentPPO.load(model_path, env=env, device='cuda')
    model = PPO.load(model_path, env=env, device='cuda')
    # model = QRDQN.load(model_path, env=env, device='cuda')
    print("Model: Model loaded...")

iters = 48
while True:	
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, log_interval=1, reset_num_timesteps=False)
	if iters == 1 or iters == 10 or iters == 50 or iters == 100 or iters == 150:
		model.save(f"{models_dir}/{model_name}_{iters}")
	model.save(f"{models_dir}/{model_name}_{iters}")
	
	print(f"Model: Model saved {iters} times.")