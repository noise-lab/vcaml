"""

    Class for initiating the teams calls in client of browser
"""

from subprocess import Popen, PIPE
from guibot.guibot import GuiBot
import time
import netrc
import sys
import pyautogui

class Teams:

    def __init__(self, args):

        self.website = args.website
        self.id = args.meeting_id
        self.profile = args.profile
        self.capture = args.capture
        self.timer = args.time
        self.host = args.host
        self.browser_type = args.browser
        
        if self.capture:
            self.filter = args.filter
            self.count = args.trials

    def launch_driver(self):

        proc = Popen(['google-chrome'])
        time.sleep(5)

        if self.browser_type == 'chrome':
            pyautogui.typewrite('chrome://webrtc-internals')
            pyautogui.hotkey('enter')
            pyautogui.hotkey('ctrl', 't')
        pyautogui.typewrite(self.id)
        pyautogui.hotkey('enter')

    def launch_app(self):
        

        print("in laucnh app")
        time.sleep(10)
        pyautogui.hotkey('tab')
        time.sleep(1)
        pyautogui.hotkey('tab')
        time.sleep(1)
        pyautogui.hotkey('tab')
        time.sleep(1)
        pyautogui.hotkey('tab')
        time.sleep(1)
        pyautogui.hotkey('enter')
        time.sleep(5)
        # guibot = GuiBot()
        # guibot.add_path('autogui_images')

        # print("in laucnh app")
        # time.sleep(10)
        # if guibot.exists('zoom_cancel_chrome.png'):
        #     guibot.click('zoom_cancel_chrome.png')


        # if guibot.exists('team_continue_browser.png'):
        #     guibot.click('team_continue_browser.png')
        # time.sleep(15)

        # if guibot.exists('teams_close_popup.png'):
        #     guibot.click('teams_close_popup.png')
        # time.sleep(5)
        # guibot.click('teams_join_meeting.png')
        # time.sleep(10)
        # guibot.click('teams_join_now.png')

        return

    def launch_client(self):

        guibot = GuiBot()
        guibot.add_path('autogui_images')

        time.sleep(5)
        if not self.host:
            time.sleep(10)

        guibot.click('zoom_open_client.png')
        time.sleep(10)
        if guibot.exists('teams_close_popup.png'):
            guibot.click('teams_close_popup.png')
        time.sleep(2)
        if self.host:
            if guibot.exists('host_join_meet.png'):
                guibot.click('host_join_meet.png')
        else:
            if guibot.exists('teams_join_meeting.png'):
                guibot.click('teams_join_meeting.png')
        time.sleep(10)
        guibot.click('teams_join_now.png')

        return

    def exit(self):

        guibot = GuiBot()
        guibot.add_path('autogui_images')

        if self.browser_type == 'chrome':
            pyautogui.hotkey('ctrl', 'tab')
            time.sleep(1)
            guibot.click('dump_webrtc.png')
            guibot.click('webrtc_download.png')
            time.sleep(1)
            pyautogui.hotkey('ctrl', 'w')
            time.sleep(1)
            pyautogui.hotkey('tab')
            time.sleep(1)
            pyautogui.hotkey('enter')
            time.sleep(5)
            # pyautogui.move(0, 100)
            # guibot.click('teams_hang_up.png')
            pyautogui.hotkey('ctrl', 'w')

        else:
            pyautogui.move(0, 100)
            guibot.click('teams_hang_up.png')
            time.sleep(3)
            res = Popen('pkill teams', shell=True)
            res = Popen('pkill chrome', shell=True)

        return

    
