"""

    Class for launching Google Meet Calls.
"""

from guibot.guibot import GuiBot
from subprocess import Popen

import time
import subprocess
import sys
import netrc
import pyautogui
import os
import signal


class Meet:

    def __init__(self, args): 

        self.website = args.website
        self.id = args.meeting_id
        self.headless = args.headless
        self.profile = args.profile
        self.capture = args.capture
        self.timer = args.time
        self.browser_type = args.browser
        self.host = args.host

        if self.capture:
            self.filter = args.filter
            self.count = args.trials

    def launch_driver(self):
        """ Function to set browser preferences and launch browser """

        # Set browser preferences

        if self.browser_type == 'firefox':
            proc = Popen(['firefox', self.id])

        elif self.browser_type == 'chrome':
            proc = Popen(['google-chrome'])
            time.sleep(5)
            pyautogui.typewrite('chrome://webrtc-internals')
            pyautogui.hotkey('enter')
            time.sleep(1)
            pyautogui.hotkey('ctrl', 't')
            time.sleep(1)
            pyautogui.typewrite(self.id)
            pyautogui.hotkey('enter')


        self.browser = proc

    def launch_app(self):
        """ Initiate a google meet video call """

        guibot = GuiBot()
        guibot.add_path('autogui_images') 

        time.sleep(15)
        print("looking for join")
        if guibot.exists('meet_join_host.png'):
            print("join found")
            guibot.click('meet_join_host.png')
        if guibot.exists('meet_ask_to_join.png'):
            guibot.click('meet_ask_to_join.png')
       
        print("moving on..")
        return

    def exit(self):

        """ Hangs up the Google Meet Call """

        guibot = GuiBot()
        guibot.add_path('autogui_images')
        
        if self.browser_type == 'chrome':
            pyautogui.hotkey('ctrl', 'tab')
            time.sleep(1)
            guibot.click('dump_webrtc.png')
            guibot.click('webrtc_download.png')
            pyautogui.hotkey('ctrl', 'w')
            time.sleep(1)
 

        guibot.click('end_meet.png')
        # guibot.click('leave_call_meet.png')
        pyautogui.hotkey('ctrl', 'w')

        return
