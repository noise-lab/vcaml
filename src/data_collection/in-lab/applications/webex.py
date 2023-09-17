from subprocess import Popen
from guibot.guibot import GuiBot
import time
import netrc
import sys
import pyautogui
pyautogui.FAILSAFE = False

class WebEx:
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
        guibot = GuiBot()
        guibot.add_path('autogui_images')

        print("in launch app")
        time.sleep(5)
        if guibot.exists('webex_1_join_meeting.png'):
            guibot.click('webex_1_join_meeting.png')

        time.sleep(5)

        if guibot.exists('webex_2_next_btn.png'):
            guibot.click('webex_2_next_btn.png')
        time.sleep(5)
        pyautogui.typewrite('vcqoe')
        pyautogui.hotkey('enter')
        time.sleep(5)
        if guibot.exists('webex_ok_btn.png'):
            guibot.click('webex_ok_btn.png')
        
        time.sleep(5)
        
        if guibot.exists('webex_3_join_meeting.png'):
            guibot.click('webex_3_join_meeting.png')
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
            if guibot.exists('webex_dl_close_btn.png'):
                guibot.click('webex_dl_close_btn.png')
            time.sleep(1)
            pyautogui.move(0, 100)
            if guibot.exists('webex_4_end_call.png'):
                guibot.click('webex_4_end_call.png')
            time.sleep(1)
            if guibot.exists('webex_5_leave_meeting.png'):
                guibot.click('webex_5_leave_meeting.png')
            time.sleep(5)
            pyautogui.hotkey('ctrl', 'w')

        else:
            pyautogui.move(0, 100)
            if guibot.exists('webex_4_end_call.png'):
                guibot.click('webex_4_end_call.png')
            time.sleep(1)
            if guibot.exists('webex_5_leave_meeting.png'):
                guibot.click('webex_5_leave_meeting.png')
            time.sleep(3)
            res = Popen('pkill chrome', shell=True)

        return
