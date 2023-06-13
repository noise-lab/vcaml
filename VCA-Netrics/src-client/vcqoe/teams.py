"""

    Class for initiating the teams calls in client of browser
"""
import os
from subprocess import Popen, PIPE

import cv2
import time
import netrc
import sys
import pyautogui
from sys import platform
import random
import string
from vcqoe.utility import click_image
import logging
log = logging.getLogger(__name__)




class Teams:

    def __init__(self, config):
        self.id = config["teams"]["url"]
        self.config = config
        self.automation_img_dir = f'{config["call"]["automation_image_dir"]}/{platform}'

    def join_call_mac(self):
        t = click_image(f'{self.automation_img_dir}/teams_close_popup.png')
        if not t:
            log.error("close popup window not found!")

        t = click_image(f'{self.automation_img_dir}/teams_continue_browser.png')
        if not t:
            log.error("error: continue on browser not found")
        time.sleep(5)

        t = click_image(f'{self.automation_img_dir}/teams_join_now.png')
        if not t:
            log.error("error: join now tab not found")
        time.sleep(3)
        
    def join_call_raspi(self):
        ## additional wait as raspi redirection takes time
        time.sleep(20)
        # joining as guest
        num_tries = 0
        max_tries = 8
        while num_tries < max_tries:
            screenshot = pyautogui.screenshot()
            screenshot.save(f'screenshot_{num_tries}.png')
            bbox = pyautogui.locate(f'{self.automation_img_dir}/teams_meeting_now.png', screenshot, confidence= 0.5, grayscale=True, region=(0, 0, 1500, 500))
            if not bbox:
                sleep_time = 5
                time.sleep(sleep_time)
                print("sleeping for {} seconds".format(sleep_time))
                num_tries += 1
            else:
                break
        if num_tries == max_tries:
            log.error("could not find enter name")
            return
        print("pressing tab key")
        pyautogui.hotkey("tab")
        time.sleep(1)
        
        print("pressing shift tab key")
        pyautogui.hotkey("shift", "tab")
        time.sleep(1)

        print("pressing shift tab key")
        pyautogui.hotkey("shift", "tab")
        
        res = ''.join(random.choices(string.ascii_uppercase +
                             string.digits, k=10))
        pyautogui.typewrite(res)

        #pyautogui.typewrite("guest")
        time.sleep(2)
        pyautogui.hotkey("tab")
        pyautogui.hotkey("enter")


    def join_call(self):
        # Open Teams VCA
        pyautogui.typewrite(self.id)
        pyautogui.hotkey('enter')
        time.sleep(2)

        if platform == "darwin":
            self.join_call_mac()
        elif platform == "linux" and os.uname()[1] == "raspberrypi":
            self.join_call_raspi()


    def exit_call(self):
        ## Hang up call
        time.sleep(1)
        pyautogui.move(0, 100)
        time.sleep(2)
        s = pyautogui.screenshot()
        s.save('screenshot.png')
        t = click_image(f'{self.automation_img_dir}/teams_leave.png', grayscale=True)
        if not t:
            print("leave image not found")
            log.error("error: leave call button not found")
        time.sleep(2)

        ## Close the browser window
        pyautogui.hotkey(self.config["platform"]["ctl_key"], 'w')

