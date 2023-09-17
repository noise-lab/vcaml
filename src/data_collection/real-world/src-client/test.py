import cv2
import numpy as np
from matplotlib import pyplot as plt
import pyautogui
from vcqoe.utility import *

for i in range(0, 5):
    print(i)
    haystack = f"screenshot_{i}.png"
    print(image_search('autogui_images/linux/teams_meeting_now_2.png', haystack=haystack, threshold=0.4))
    print('-------')
