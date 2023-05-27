import cv2
import numpy as np
import time


def render_frame(frame):                #📍 This is just for debugging purposes. It renders a cv2 image. Use it to make sure your game is being captured correctly.
    cv2.imshow('debug-render', frame)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
 
"""
HP_CHART = {}
#📍  saving the vigor chart from the csv file into variables
with open('vigor_chart.csv', 'r') as v_chart:
    for line in v_chart.readlines():
        stat_point = int(line.split(',')[0])
        hp_amount = int(line.split(',')[1])
        HP_CHART[stat_point] = hp_amount
"""


#📝 To do:
#📝 Implement the vigor-hp csv file and make sure it works with the hp bar detection
#📝 Same for stamina
#📝 Finally fix the health bar reading. Computer vision is weird...


class EldenReward:
    #📍 Constructor
    def __init__(self) -> None:
        #self.previous_runes_held = None
        #self.current_runes_held = None
        #self.seen_boss = False
        #self.time_since_seen_boss = time.time()
        #self.hp_history = []

        # self.max_hp = 499
        self.max_hp = 455
        self.prev_hp = 1.0     
        self.curr_hp = 1.0
        self.time_since_dmg_taken = time.time()
        self.death = False
        self.curr_stam = 1.0
        
        #📍 2 Boss variables
        self.curr_boss_hp = 1.0
        self.prev_boss_hp = 1.0
        self.min_boss_hp = 1.0
        self.time_since_boss_dmg = time.time() 
        self.boss_death = False        

        #📍 3 Other
        self.image_detection_tolerance = 0.02          #📍 The image detection of the hp bar is not perfect.

    #📍 Methods
    def get_current_hp(self, frame):
        #self.rewardGen.max_hp = 100
        hp_ratio = 0.403
        hp_image = frame[65:70, 205:290 + int(self.max_hp * hp_ratio) - 20]
        # render_frame(hp_image)
        lower = np.array([0,90,75])
        upper = np.array([150,255,125])
        hsv = cv2.cvtColor(hp_image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        # render_frame(mask)
        matches = np.argwhere(mask==255)
        curr_hp = len(matches) / (hp_image.shape[1] * hp_image.shape[0])

        curr_hp += 0.02
        if curr_hp >= 0.96:
            curr_hp = 1.0

        return curr_hp

    def get_current_stamina(self, frame):
        stam_image = frame[115:120, 200:200 + 445]
        # render_frame(stam_image)
        lower = np.array([0,100,0])
        upper = np.array([150,255,150])
        hsv = cv2.cvtColor(stam_image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        # render_frame(mask)
        matches = np.argwhere(mask==255)
        self.curr_stam = len(matches) / (stam_image.shape[1] * stam_image.shape[0])

        self.curr_stam += 0.02
        if self.curr_stam >= 0.96:
            self.curr_stam = 1.0
        #print('🏃 Stamina: ', self.curr_stam)
        return self.curr_stam
    

    def get_boss_hp(self, frame):
        boss_hp_image = frame[1160:1165, 623:1950]
        # render_frame(boss_hp_image)
        lower = np.array([0,130,0])
        upper = np.array([255,255,255])
        hsv = cv2.cvtColor(boss_hp_image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        # render_frame(mask)
        matches = np.argwhere(mask==255)
        boss_hp = len(matches) / (boss_hp_image.shape[1] * boss_hp_image.shape[0])
        self.min_boss_hp = min(self.min_boss_hp, boss_hp)
        # print('👹 Boss HP: ', boss_hp)

        return boss_hp
 
    def update(self, frame):
        self.curr_boss_hp = self.get_boss_hp(frame)
        self.curr_hp = self.get_current_hp(frame)
        self.curr_stam = self.get_current_stamina(frame)
        
        self.death = False
        if self.curr_hp <= 0.01 + self.image_detection_tolerance:
            self.death = True
            self.curr_hp = 0.0

        self.boss_death = False
        if self.curr_boss_hp <= 0.01:
            self.boss_death = True

        #📍 1. Hp Rewards
        hp_reward = 0
        if not self.death:                                                         
            if self.curr_hp >= self.prev_hp + self.image_detection_tolerance:
                hp_reward = 10                  
            elif self.curr_hp <= self.prev_hp - self.image_detection_tolerance:
                hp_reward = -40
                self.time_since_dmg_taken = time.time()
            self.prev_hp = self.curr_hp
        else:
            hp_reward = -125

        time_since_taken_dmg_reward = 0                                    
        if time.time() - self.time_since_dmg_taken > 7:
            time_since_taken_dmg_reward = 20

        #📍 2. Boss Rewards
        boss_dmg_reward = 0
        if self.boss_death:
            boss_dmg_reward = 400
        else:
            if self.curr_boss_hp < self.prev_boss_hp: #- self.image_detection_tolerance  + 0.01:
                print('👹 Boss took damage', (self.prev_boss_hp - self.curr_boss_hp) * 100)
                boss_dmg_reward = 25 + (self.prev_boss_hp - self.curr_boss_hp) * 100
                self.time_since_boss_dmg = time.time()
            if time.time() - self.time_since_boss_dmg > 10:
                boss_dmg_reward = -50
            elif time.time() - self.time_since_boss_dmg > 5:
                boss_dmg_reward = -25
        self.prev_boss_hp = min(self.min_boss_hp, self.prev_boss_hp)
        # print(self.curr_boss_hp, self.prev_boss_hp, self.min_boss_hp)

        percent_through_fight_reward = 0
        if self.min_boss_hp < 0.97:
            percent_through_fight_reward = (1-self.min_boss_hp) * 200 


        #📍 3. Other Rewards
        """
        dodge_reward = 0
        #dodge reward will be hard to implement if we dont just want the agent to spam dodge. So this will be on hold for now

        boss_found_reward = 0
        #maybe for open world bosses?

        
        time_alive_reward = 0
        #time alive reward will be hard to implement if we dont just want the agent to run away and survive. So this will be on hold for now
        """

        total_reward = hp_reward + boss_dmg_reward + time_since_taken_dmg_reward + percent_through_fight_reward #+ dodge_reward + boss_found_reward + time_alive_reward
        total_reward = round(total_reward, 3)

        return total_reward, self.death, self.boss_death