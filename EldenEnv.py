import os
import cv2
import gym
import time
import numpy as np
from gym import spaces
import mss
from EldenReward import EldenReward
import pydirectinput
import pytesseract                          #📍 This is used to read the text on the screen
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' #📍path to pytesseract. We need it for image to string conversion

import walkToBoss         #📍 This is the function that walks from the grace to the boss. These are hard coded for every boss and need to be changed if you want to fight a different boss.

print("EldenEnv.py #0")

#📝 To do:
#📝 0. 
#📝 1. We need to be able to set our vigor stat somewhere. And the hp bar detection needs to be based on that. (in EldenReward)
    #📝 1.1 Implement the vigor-hp csv file and make sure it works with the hp bar detection (how much hp the player has based on his vigor (how long the ho bar is))   (in EldenReward)
#📝 2 Finally fix the health bar reading. Computer vision is weird... (in EldenReward)
#📝 3. Tensorboard visualization (in train.py)

N_CHANNELS = 3
IMG_WIDTH = 1920
IMG_HEIGHT = 1080
MODEL_WIDTH = int(800 / 2)
MODEL_HEIGHT = int(450 / 2)


DISCRETE_ACTIONS = {'release_wasd': 'release_wasd', #📍 All the action the agent can take (just a list to count them. This isnt used anywhere)
                    'w': 'run_forwards',                
                    's': 'run_backwards',
                    'a': 'run_left',
                    'd': 'run_right',
                    'shift': 'dodge',
                    'u': 'attack',
                    'i': 'strong_attack',
                    'o': 'magic',
                    'e': 'use_item'}

ACTION_NUM_TO_WORD = {
    0: 'Release',
    1: 'Forwards',
    2: 'Backwards',
    3: 'Left',
    4: 'Right',
    5: 'Dodge',
    6: 'Attack',
    7: 'Shield',
    8: 'Jump',
    9: 'Use Item'
}

N_DISCRETE_ACTIONS = len(DISCRETE_ACTIONS)
NUM_ACTION_HISTORY = 10

def oneHotPrevActions(actions):
    oneHot = np.zeros(shape=(NUM_ACTION_HISTORY, N_DISCRETE_ACTIONS))
    if not actions:
        return oneHot.reshape((NUM_ACTION_HISTORY, N_DISCRETE_ACTIONS, 1))
    
    for i in range(NUM_ACTION_HISTORY):
        if len(actions) >= (i + 1):
            oneHot[i][actions[-(i + 1)]] = 1
    return oneHot.reshape((NUM_ACTION_HISTORY, N_DISCRETE_ACTIONS, 1))

#🐜 Just for debugging purposes. It renders a cv2 image. Use it to make sure any screenshots are correct.
def render_frame(frame):                
    cv2.imshow('debug-render', frame)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()
        
#📍 The is the actual environment.
class EldenEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    #📍 The constructor of the class. This is where we initialize the environment.
    def __init__(self):
        super(EldenEnv, self).__init__()                        #📍 something about initializing the class correctly. I dont know but its required.
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS) #📍 action space that the agent can take. (0-9) (this is used in train.py)

        spaces_dict = {                                         #📍 observation space that the agent can see. (img, prev_actions, state)
            'img': spaces.Box(low=0, high=255, shape=(MODEL_HEIGHT, MODEL_WIDTH, N_CHANNELS), dtype=np.uint8),
            'prev_actions': spaces.Box(low=0, high=1, shape=(NUM_ACTION_HISTORY, N_DISCRETE_ACTIONS, 1), dtype=np.uint8),
            'state': spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32),
        }

        self.observation_space = gym.spaces.Dict(spaces_dict)  #📍 Defining the observation space for gym. (this is used in train.py)
        
        #📍 Class variables
        self.reward = 0                         #📍 Current reward
        self.rewardGen = EldenReward()          #📍 Setting up the reward generator class (see EldenReward.py)
        self.death = False                      #📍 If the agent died
        self.t_start = time.time()              #📍 Time when the game started
        self.done = False                       #📍 If the game is done
        self.iteration = 0                      #📍 Current iteration (number of steps taken in this fight)
        self.backprop_iteration = 1000           #📍 Current iteration (number of steps taken in this fight)
        self.total_iterations = 0               #📍 Total number of iterations (number of steps taken in all fights)
        self.first_step = False                 #📍 If this is the first step (is set to true in reset)
        #self.locked_on = False                 #📍 Log on needs to be hardcoded for now. (in walkToBoss.py)
        self.max_reward = None                  #📍 The maximum reward that the agent has gotten
        self.reward_history = []                #📍 array of the rewards to calculate the average reward of fight            
        self.sct = mss.mss()                    #📍 initializing CV2 and MSS (used to take screenshots)
        #self.boss_hp_end_history = []          #📍 array of the boss hp at the end of each run (not implemented)
        self.action_history = []                #📍 array of the actions that the agent took. (see oneHotPrevActions and the observation space)
        self.time_since_heal = time.time()      #📍 time since the last heal
        

    #📍 Grabbing the screenshot of the game
    def grab_screen_shot(self):
        for _, monitor in enumerate(self.sct.monitors[2:], 1):
            sct_img = self.sct.grab(monitor)    # Get get screenshot of whole screen
            frame = cv2.cvtColor(np.asarray(sct_img), cv2.COLOR_BGRA2RGB)
            # frame = frame[46:IMG_HEIGHT + 46, 12:IMG_WIDTH + 12]    #cut the frame to the size of the game
            # render_frame(frame)    #🐜 render the frame for debugging
            # print('📷 screenshot grabbed')
            return frame
    
    #📍 Taking an action in the game using pydirectinput
    def take_action(self, action):
        #action = -1 #🐜 Do not take any action
        # Cancel all actions
        if action == 0:
            pydirectinput.keyUp('w')
            pydirectinput.keyUp('s')
            pydirectinput.keyUp('a')
            pydirectinput.keyUp('d')
            pydirectinput.mouseUp(button=pydirectinput.SECONDARY) 
            pydirectinput.mouseUp(button=pydirectinput.PRIMARY)
        # move up
        elif action == 1:
            pydirectinput.keyUp('w')
            pydirectinput.keyUp('s')
            pydirectinput.keyDown('w')
        # move down
        elif action == 2:
            pydirectinput.keyUp('w')
            pydirectinput.keyUp('s')
            pydirectinput.keyDown('s')
        # move left
        elif action == 3:
            pydirectinput.keyUp('a')
            pydirectinput.keyUp('d')
            pydirectinput.keyDown('a')
        # Move right
        elif action == 4:
            pydirectinput.keyUp('a')
            pydirectinput.keyUp('d')
            pydirectinput.keyDown('d')
        # dodge
        elif action == 5:
            pydirectinput.press(' ')
        # Primary Attack
        elif action == 6:
            pydirectinput.mouseUp(button=pydirectinput.SECONDARY) 
            pydirectinput.leftClick(duration=0.1)
        # Secondary Attack
        elif action == 7:
            pydirectinput.mouseUp(button=pydirectinput.PRIMARY) 
            pydirectinput.rightClick(duration=0.1)
        # Jump
        elif action == 8:
            pydirectinput.press('f')
        # item
        elif action == 9 and time.time() - self.time_since_heal > 1.5 and self.rewardGen.curr_hp < 0.8:
            pydirectinput.press('r')        
            self.time_since_heal = time.time()
            return
        
    def step(self, action):
        self.total_iterations += 1
        pre_backprop = False

        if self.first_step:
            print("🐾#1 first step")
            self.death = False
            self.done = False
        
        t0 = time.time()
        
        frame = self.grab_screen_shot()
        if self.first_step and self.rewardGen.get_boss_hp(frame) < 0.5:
            walkToBoss.walk_to_godrick()

        self.reward, self.death, self.boss_death = self.rewardGen.update(frame)

        if self.first_step:
            self.reward = 0

        # print(self.total_iterations, self.backprop_iteration)
        if self.total_iterations > 0 and self.total_iterations == self.backprop_iteration:
            print('Model: (pre-backprop iteration movement stopper)')
            pre_backprop = True
            self.take_action(0)

        if self.total_iterations > 0 and self.total_iterations - 1 == self.backprop_iteration:
            print('Model: (post-backprop iteration movement normalizer)')
            self.take_action(0)
            self.total_iterations = 1
            self.done = True
            self.reward = 0
            
        if not self.death:
            if (time.time() - self.t_start) > 600:
                print('Failure: taking too long, giving up...')
                self.take_action(0)
                self.done = True
                print('Model: Step done (time limit)')
            elif self.boss_death:
                print('Model: Step done (boss dead)')                                                            
                self.done = True
            #📝elif more conditions to end the step loop here:
                #📝 1 Boss is lost (for open world bosses maybe)
                #📝 2 ...idk
        #📍 Player death
        else:
            print('Model: Step done (death)') 
            self.done = True
        
        if not self.done and not pre_backprop:
            self.take_action(action)
        
        observation = cv2.resize(frame, (MODEL_WIDTH, MODEL_HEIGHT))
        #render_frame(observation)
        info = {}
        self.first_step = False
        self.iteration += 1

        if self.max_reward is None:
            self.max_reward = self.reward
        elif self.max_reward < self.reward:
            self.max_reward = self.reward

        self.reward_history.append(self.reward)

        #FPS LIMITER
        t_end = time.time()                                             
        # desired_fps = (1 / 60)
        # time_to_sleep = desired_fps - (t_end - t0)

        #print(1 / (time.time() - t0))
        # if time_to_sleep > 0:
        #     time.sleep(time_to_sleep)
        #END FPS LIMITER
        current_fps = round(((1 / (t_end - t0)) * 10), 0) 

        #📍 5. This is the actual observation that we return
        spaces_dict = {
            'img': observation,
            'prev_actions': oneHotPrevActions(self.action_history),
            'state': np.asarray([self.rewardGen.curr_hp, self.rewardGen.curr_stam, self.rewardGen.min_boss_hp])
        }
        
        self.action_history.append(int(action))

        if not self.done:
            print(f'Model: Iteration: {self.iteration} | FPS: {current_fps} | Reward: {self.reward} | Max Reward: {self.max_reward} | Action: {ACTION_NUM_TO_WORD[action]}')
        else:
            print(f'Model: Reward: {self.reward} | Max Reward: {self.max_reward}')
            self.take_action(0)

        return spaces_dict, self.reward, self.done, info
    
    def reset(self):
        print('Model: reset called...')
        self.take_action(0)
        print('Model: Unholding keys...')

        if len(self.reward_history) > 0:
            total_r = np.sum(self.reward_history)
            avg_r = total_r / len(self.reward_history)                              
            print('Model: Average reward for last run:', avg_r) 

        time.sleep(2)

        t_check_frozen_start = time.time()
        loading_screen_flag = False
        t_since_seen_next = None
        while True:
            frame = self.grab_screen_shot()
            next_text_image = frame[1340:1400, 175:315]
            next_text_image = cv2.resize(next_text_image, ((205-155)*3, (1040-1015)*3))
            lower = np.array([0,0,75])    
            upper = np.array([255,255,255])
            hsv = cv2.cvtColor(next_text_image, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            #matches = np.argwhere(mask==255)
            #percent_match = len(matches) / (mask.shape[0] * mask.shape[1])
            #print(percent_match)       #📍 Percentage of white pixels in the mask
            next_text = pytesseract.image_to_string(mask,  lang='eng',config='--psm 6 --oem 3') 
            loading_screen = "Next" in next_text or "next" in next_text

            if loading_screen:
                print("Loading Screen:", loading_screen)
                loading_screen_flag = True
                t_since_seen_next = time.time()
            else:
                if loading_screen_flag:
                    print('Loading screen seen. Walk back to boss will start in 2.5 seconds...')
                else:
                    print('No loading screen. Step loop will start in 30 seconds or after loading screen.')
                
            if not t_since_seen_next is None and ((time.time() - t_check_frozen_start) > 7.5) and (time.time() - t_since_seen_next) > 2.5:  #📍 We were in a loading screen and left it. (Start step after 2.5 seconds not seeling a loading screen)
                print('Model: Left loading screen #1')
                break
            elif t_since_seen_next is None and ((time.time() - t_check_frozen_start) > 30):
                print('Model: No loading screen found #2')
                break
            #📝 elif any of the other conditions are met:
                #📝 we could do something like staying in this loop until we see a full boss health bar then press the lock on key and start the next step loop. this way we would have automatic initiation of the next step.
                #📝 If we then wait forever in this loop until that happens, we could just leave the game running and it would automatically start the next step when we enter a boss arena.
            #📝 elif ny of the other conditions are met:
                #📝 or you could aput in a manual break condition here that you set in a different thread. Maybe if you want to use your computer while the game is running you could set a break condition that you can set with a hotkey.
                #📝 if you also set self.done = True here, the environment will reset and stop moving the character.
        
        print("🔄👹 walking to boss")
        walkToBoss.walk_to_godrick()

        self.iteration = 0
        self.reward_history = [] 
        self.done = False
        self.first_step = True
        #self.locked_on = False                             #✂️ Unused
        self.max_reward = None
        #self.rewardGen.seen_boss = False                   #✂️ Maybe for open world bosses?
        #self.rewardGen.time_since_seen_boss = time.time()  #✂️ Unused
        self.rewardGen.prev_hp = 1
        self.rewardGen.curr_hp = 1
        #self.rewardGen.time_since_reset = time.time()      #✂️ Unused
        #self.rewardGen.time_since_dmg_healed = time.time() #✂️ Unused
        self.rewardGen.time_since_dmg_taken = time.time()
        #self.rewardGen.hits_taken = 0                      #✂️ Unused
        self.rewardGen.curr_boss_hp = 1 
        self.rewardGen.prev_boss_hp = 1
        self.rewardGen.min_boss_hp = 1
        self.t_start = time.time()


        observation = cv2.resize(frame, (MODEL_WIDTH, MODEL_HEIGHT))
        self.action_history = []
        spaces_dict = { 
            'img': observation,
            'prev_actions': oneHotPrevActions(self.action_history),
            'state': np.asarray([1.0, 1.0, 1.0])                         #📍 Full hp, stamina, boss health
        }
        
        print('Model: Reset done.')
        return spaces_dict 

    def render(self, mode='human'):
        pass

    def close (self):
        self.cap.release()





