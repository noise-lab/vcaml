"""

    Class for launching Google Meet Calls.
"""
from vcqoe.utility import *
from subprocess import Popen

import time
import subprocess
import sys
import netrc
import pyautogui
import os
import signal
import cv2
import random
import string

from sys import platform
class Meet:

    def __init__(self, config):
        self.id = config["meet"]["url"]
        self.config = config
        self.automation_img_dir = f'{config["call"]["automation_image_dir"]}/{platform}'

    def join_call_mac(self):
        click_image(f'{self.automation_img_dir}/meet_join_now.png')

    def join_call_raspi(self):
        time.sleep(10)
        res = ''.join(random.choices(string.ascii_uppercase +
                             string.digits, k=10))
        pyautogui.typewrite(res)
        time.sleep(1)
        pyautogui.hotkey("tab")
        pyautogui.hotkey("enter")

    def join_call(self):

        pyautogui.typewrite(self.id)
        pyautogui.hotkey('enter')
        time.sleep(5)
        
        if platform == "darwin":
            self.join_call_mac()
        elif platform == "linux" and os.uname()[1] == "raspberrypi":
            self.join_call_raspi()


        
    def exit_call(self):
        ## End Meet call
        click_image(f'{self.automation_img_dir}/meet_end_call.png')
        time.sleep(2)
        
        pyautogui.hotkey(self.config["platform"]["ctl_key"], 'w')
        return
