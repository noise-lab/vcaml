""" vcqoe command-line interface entry-point """

import argparse
import subprocess
import threading
import time
import pyautogui
import toml
from subprocess import Popen
from vcqoe.meet import Meet
from vcqoe.vca import VCA
from vcqoe.teams import Teams
from vcqoe.capture import CaptureTraffic
from vcqoe.utility import *
from cv2 import *


from vcqoe.utility import get_trace_filename
import logging
import logging.config
import sys



def execute(config_file=None):
    """ Execute the auto_call CLI command """
    try:

        config = toml.load(config_file)
        datapath = config["default"]["datapath"]
        vca_name = config["default"]["vca"]
        capture = config["default"]["capture"]
        trace_filename = get_trace_filename(config)

        logging.basicConfig(filename=f'{trace_filename}.log', encoding='utf-8', level=logging.DEBUG)

        if capture:
            capture_thread = CaptureTraffic(config, trace_filename)
            capture_thread.start()

        rtc_dir = f'{datapath}/{vca_name}/webrtc'
        make_dir(rtc_dir)

        vca = VCA(config, trace_filename)
        
        vca.start_call()
        time.sleep(confip["default"]["call_duration"])
        vca.end_call()

        if capture:
            capture_thread.is_running = False

    except KeyboardInterrupt:
        print('\n KEYBOARD INTERRUPT...CLOSED \n')


def build_parser():
    """ Construct parser to interpret command-line args """

    parser = argparse.ArgumentParser(
        description='Initiate and capture video calls')

    parser.add_argument(
        'website',
        help="Website to visit. Currently supports [meet] and [teams]",
    )

    parser.add_argument(
        '-e', '--experiment_name',
        action='store',
        help='Name of experiment (used to name dir with captures)'
    )

    parser.add_argument(
        '-tr', '--trace',
        action='store',
        nargs=7,
        help='Trace name for saving captures in agreed format: [speed std_speed latency std_latency loss std_loss timestamp]'
    )

    parser.add_argument(
        '-m', '--metrics',
        default=False,
        action='store_true',
        help='Collect QoE metrics for instances at [meeting]'
    )

    parser.add_argument(
        '-id', '--meeting_id',
        default=None,
        help="URL for meeting"
    )

    parser.add_argument(
        '-host', '--host',
        default=False,
        action='store_true',
        help='Set instance to host (does not leave call)'
    )

    parser.add_argument(
        '-p', '--profile',
        default=None,
        help="Directory of browser profile"
    )

    parser.add_argument(
        '-c', '--capture',
        action="store_true",
        default=False,
        help="Set flag to capture network traffic"
    )

    parser.add_argument(
        '-f', '--filter',
        default="",
        help="lipcap filter for capture"
    )

    parser.add_argument(
        '-n', '--trials',
        type=int,
        default=1,
        help="Number of captures (default is 1)"
    )

    parser.add_argument(
        '-t', '--time',
        type=int,
        default=60,
        help="Length (secs) of capture. (default is 60)"
    )

    parser.add_argument(
        '-i', '--headless',
        action="store_true",
        help="Run headless browser",
    )

    parser.add_argument(
        '-video', '--video-file',
        type=str,
        default='~/vca/rush_hour_1080p25.y4m',
        help="Name of the video file to be played at the sender"
    )

    parser.add_argument(
        '-loop', '--loop',
        type=str,
        default='False',
        action='store_true',
        help="Enable looping of video at the sender. If set to false, the video loops for the full duration of the call."
    )

    parser.add_argument(
        '-sender', '--sender',
        default='192.168.1.166',
        help = 'Private IP address of the sender. Requires an SSH key setup.'
    )
    return parser




