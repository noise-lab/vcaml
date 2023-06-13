"""

    Class for initiating zoom calls in browser or in client
"""

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver import Firefox
from selenium.webdriver.firefox.options import Options
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from subprocess import Popen, PIPE
from guibot.guibot import GuiBot
from pathlib import Path
import time
import netrc
import sys
import pyautogui


class Zoom:

    def __init__(self, args):

        self.website = args.website
        self.id = args.meeting_id
        self.headless = args.headless
        self.profile = args.profile
        self.capture = args.capture
        self.timer = args.time
        self.host = args.host
        self.browser_type = args.browser

        if self.capture:
            self.filter = args.filter
            self.count = args.trials

    def launch_driver(self):
        """ Function to set browser preferences and launch browser """

        # Set browser preferences
        opts = Options()

        opts.set_preference("permissions.default.microphone", 1)
        opts.set_preference("permissions.default.camera", 1)
        opts.set_preference("permissions.default.desktop-notification", 0)
        opts.set_preference("dom.webnotifications.enabled", False)
        opts.set_headless(headless=self.headless)

        if self.browser_type == 'firefox':
            fp = webdriver.FirefoxProfile(self.profile)

            cap = DesiredCapabilities().FIREFOX

            self.browser = webdriver.Firefox(capabilities=cap, options=opts,
                                             firefox_profile=fp)
            self.browser.maximize_window()
        else:
            opts = webdriver.ChromeOptions()
            opts.add_argument(f'--user-data-dir={self.profile}')
            opts.set_headless(headless=self.headless)

            self.browser = webdriver.Chrome(options=opts)
        self.browser.implicitly_wait(10)

    def launch_client(self):
        time.sleep(10)
        self.browser.get(self.id)
        time.sleep(5)
        guibot = GuiBot()
        guibot.add_path('autogui_images')

        guibot.click('zoom_open_client.png')
        time.sleep(15)

        if guibot.exists('zoom_join_with_video.png'):
            guibot.click('zoom_join_with_video.png')
        time.sleep(15)
        guibot.click('zoom_join_audio_client.png')

        return

    def launch_app(self):
        """
        Function that launches a zoom meeting through the UChicago login

        """

        if self.browser_type == 'chrome':
            self.browser.get("chrome://webrtc-internals")
            self.browser.execute_script(
                    "window.open('https://zoom.us/google_oauth_signin', '_blank')")
            time.sleep(2)
            self.browser.switch_to_window(self.browser.window_handles[1])
        else:
            self.browser.get("https://zoom.us/google_oauth_signin")
            time.sleep(5)


        if self.id is None:
            sys.exit("Insert zoom meeting ID...Quitting")

        self.browser.get(self.id + "#success")
        time.sleep(15)
        guibot = GuiBot()
        guibot.add_path('autogui_images')

        guibot.click('zoom_launch_meeting.png')

        if self.browser_type == 'chrome':
            guibot.click('zoom_cancel_chrome.png')
        else:
            guibot.click('zoom_firefox_cancel.png')

        try:
            link = self.browser.find_element_by_link_text('Join from Your Browser')
            link.click()
        except NoSuchElementException:
            print('No join by browser')

        if guibot.exists('zoom_join_firefox_browser.png'):
            guibot.click('zoom_join_firefox_browser.png')
        time.sleep(15)

        if guibot.exists('zoom_join_audio_browser.png'):
            guibot.click('zoom_join_audio_browser.png') 

        return

    def exit(self):
    
        guibot = GuiBot()
        guibot.add_path('autogui_images')

        if self.browser_type == 'chrome':
            self.browser.switch_to_window(self.browser.window_handles[0])
            guibot.click('dump_webrtc.png')
            self.browser.find_element_by_xpath(
                    '//*[@id="content-root"]/details/div/div[1]/a/button').click()
            time.sleep(5)
            window_name = self.browser.window_handles[1]
            self.browser.switch_to_window(window_name)

        pyautogui.move(0, 100)
       
        if self.host:
            guibot.add_path('autogui_images/end_zoom.png')
            guibot.add_path('autogui_images/end_all_zoom.png')
            if guibot.exists('autogui_images/end_zoom.png'):
                guibot.click('autogui_images/end_zoom.png')
            else:
                print("Can't find exit button (host)")
            time.sleep(1)

            if guibot.exists('autogui_images/end_all_zoom.png'):
                guibot.click('autogui_images/end_all_zoom.png')
            else:
                print("Can't find end all button (host)")

        else:
            if self.browser_type == 'chrome' or self.browser_type == 'firefox':
                guibot.click('zoom_leave_browser.png')
                guibot.click('leave_meeting_zoom_browser.png')
            else:
                pyautogui.move(100, 100)
                guibot.click('leave_zoom.png')
                guibot.click('zoom_leave_meeting_client.png')
            self.browser.quit()

        return

    def login(self):
        """ Standard login keystrokes """

        guibot = GuiBot()
        guibot.add_path('autogui_images')


        print('logging')

        actions = ActionChains(self.browser)
        actions = actions.send_keys(self.username)
        actions = actions.send_keys(Keys.TAB)
        actions = actions.send_keys(self.password)
        actions = actions.send_keys(Keys.ENTER)
        actions.perform()
        time.sleep(5)
