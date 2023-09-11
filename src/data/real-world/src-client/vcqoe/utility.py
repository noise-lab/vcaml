'''
Contains generic utility functions
'''
import  pathlib
import subprocess
import time
import pyautogui
import cv2
import numpy as np

def make_dir(new_dir):
    """
    Creates a directory if it does not exists including the
    parent directories
    @param new_dir:
    @return:
    """
    p = pathlib.Path(new_dir)
    if not p.is_dir():
        print('making dir')
        p.mkdir(parents=True)
    return


def image_search(image_name, haystack = None, threshold=0.8, grayscale=False):
    if haystack:
        sct = cv2.imread(haystack)
    else:
        sct = pyautogui.screenshot()
    img_rgb = np.array(sct)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    img = img_gray if grayscale else img_rgb
    template = cv2.imread(image_name, 0)
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print(max_val)
    if max_val < threshold:
        return [-1, -1]
    else:
        return max_loc


def click_image(png_name, grayscale=False, confidence=0.9):
    scale_factor = 2
    button = pyautogui.locateCenterOnScreen(png_name, grayscale=grayscale, confidence=confidence)
    if not button:
        return None
    x = button[0]
    y = button[1]
    if subprocess.call("system_profiler SPDisplaysDataType | grep 'Retina'", shell= True) == 0:
        x = x / scale_factor
        y = y / scale_factor
    pyautogui.moveTo(x,y, 1)
    pyautogui.click()
    return "Clicked {0}".format(png_name)


def get_trace_filename(config):
    datapath = config["default"]["datapath"]
    vca_name = config["default"]["vca"]
    data_dir = f'{datapath}/{vca_name}/'
    make_dir(data_dir)
    trace_filename = f"{data_dir}/{vca_name}-{int(time.time())}"
    return trace_filename
