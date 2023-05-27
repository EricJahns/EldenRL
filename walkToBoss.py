import pydirectinput
import time

class Navigation():
    def __init__(self):
        self.FORWARDS = 'w'
        self.LEFT = 'a'
        self.RIGHT = 'd'
        self.BACKWARDS = 's'
        self.JUMP = ' '
        self.LOCK_ON = 'q'
        self.SELECT = 'e'
        self.BOSS = None

    def walk_to_margit(self):
        self.BOSS = "Margit"
        print("Model: walking #0 to the fog gate")
        pydirectinput.keyDown('shift')
        pydirectinput.keyDown(self.FORWARDS)
        pydirectinput.keyDown(self.RIGHT)
        time.sleep(0.5)
        pydirectinput.keyUp(self.RIGHT)
        time.sleep(2.5)
        pydirectinput.keyDown(self.RIGHT)
        time.sleep(0.7)
        pydirectinput.keyUp(self.RIGHT)
        time.sleep(3.3)
        pydirectinput.press(self.SELECT)
        time.sleep(2)
        print("walking #1 lock on to the boss")
        pydirectinput.keyDown(self.FORWARDS)
        time.sleep(0.8)
        pydirectinput.keyUp(self.FORWARDS)
        pydirectinput.press(self.LOCK_ON)
        print("walking done")

    def walk_to_godrick(self):
        self.BOSS = "Godrick"
        print("Model: walking #1 to the fog gate")
        pydirectinput.press(self.LOCK_ON)
        time.sleep(0.1)
        pydirectinput.keyDown(self.FORWARDS)
        time.sleep(2)
        pydirectinput.keyDown(self.RIGHT)
        time.sleep(2)
        pydirectinput.keyUp(self.RIGHT)
        time.sleep(0.5)
        pydirectinput.keyUp(self.FORWARDS)
        pydirectinput.press(self.SELECT)
        time.sleep(3)
        print("Locking onto the boss")
        pydirectinput.keyDown(self.FORWARDS)
        time.sleep(3.5)
        pydirectinput.keyUp(self.FORWARDS)
        time.sleep(0.5)
        pydirectinput.press(self.LOCK_ON)
        print("Walking Done")

    def walk_to_soldier_godrick(self):
        self.BOSS = "Soldier Godrick"
        print("Model: walking #1 to the fog gate")
        pydirectinput.press(self.LOCK_ON)
        time.sleep(0.1)
        pydirectinput.keyDown(self.FORWARDS)
        time.sleep(2)
        pydirectinput.keyDown(self.RIGHT)
        time.sleep(2)
        pydirectinput.keyUp(self.RIGHT)
        time.sleep(0.5)
        pydirectinput.keyUp(self.FORWARDS)
        pydirectinput.press(self.SELECT)
        time.sleep(3)
        print("Locking onto the boss")
        pydirectinput.keyDown(self.FORWARDS)
        time.sleep(3.5)
        pydirectinput.keyUp(self.FORWARDS)
        time.sleep(0.5)
        pydirectinput.press(self.LOCK_ON)
        print("Walking Done")