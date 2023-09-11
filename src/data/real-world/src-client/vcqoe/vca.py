import threading
import time
from vcqoe.meet import Meet
from vcqoe.teams import Teams
from vcqoe.utility import *
from subprocess import Popen
import pyautogui
from sys import platform
import os
import Xlib.display
class VCA():

    vca_client = None

    def __init__(self, config, filename):
        self.config = config

        self.vca_name = config["default"]["vca"]
        if self.vca_name == "meet":
            self.vca_client = Meet(self.config)
        elif self.vca_name == "teams":
            self.vca_client = Teams(self.config)

        self.json_fname = f'{filename}.json'
        self.automation_img_dir = f'{config["call"]["automation_image_dir"]}/{platform}'
        self.DISPLAY = ':0'

    def start_virtual_display(self):
        #self.stop_virtual_display()
        self.DISPLAY = ':99'
        cmd = f"nohup Xvfb {self.DISPLAY} -screen 0 1920x1080x24 2> nohup.out &"
        print(cmd)
        Popen(cmd, close_fds=True, shell=True)
        time.sleep(1)
        #print("Starting virtual display")
        os.environ['DISPLAY'] = self.DISPLAY
        pyautogui._pyautogui_x11._display = Xlib.display.Display(os.environ["DISPLAY"])

    def stop_virtual_display(self):
        _ = Popen('killall Xvfb', shell=True)

    def launch_browser(self):
        flags = ""
        if self.config["default"]["virtual"]:
            flags = "--disable-gpu"
        proc = Popen(f'{self.config["platform"]["browserpath"]} {flags}', shell=True)
        time.sleep(4)

        # Opening WebRTC internals
        pyautogui.typewrite('chrome://webrtc-internals')
        pyautogui.hotkey('enter')
        pyautogui.hotkey(self.config["platform"]["ctl_key"], 't')

    
    def dump_webrtc(self):
        ## Download and close webrtc tab
        # Remove any prior webrtc files in the download directory
        os.system("rm ~/Downloads/webrtc-internals*")
        pyautogui.hotkey(self.config["platform"]["ctl_key"], '1')
        time.sleep(1)
        click_image(f'{self.automation_img_dir}/dump_webrtc.png')
        time.sleep(2)
        click_image(f'{self.automation_img_dir}/webrtc_download.png')
        time.sleep(2)
        pyautogui.hotkey(self.config["platform"]["ctl_key"], 'w')


    def start_call(self):
        if self.config["default"]["virtual"]:
            self.start_virtual_display()

        self.launch_browser()
        self.vca_client.join_call()

        
    def end_call(self):
        # download webrtc stats from browser
        self.dump_webrtc()

        # end the call and close the browser
        self.vca_client.exit_call()

        # copy webrtc stats to data directory
        self.copy_webrtc_stats()
        
        # stop virtual display
        if self.config["default"]["virtual"]:
            self.stop_virtual_display()


    def copy_webrtc_stats(self):
        print(self.json_fname)

        _ = Popen(f'cp ~/Downloads/webrtc_internals_dump.txt {self.json_fname}', shell=True)
        _ = Popen('rm ~/Downloads/webrtc_internals_dump.txt', shell=True)

        return
