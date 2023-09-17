""" vcqoe command-line interface entry-point """
from threading import Thread
import argparse
import pathlib
import time
from subprocess import Popen, PIPE, STDOUT
from applications.meet import Meet
from applications.zoom import Zoom
from applications.teams import Teams
from applications.webex import WebEx
from zoom_api.metrics import ZoomMetrics
from capture import capture_traffic
import os
import paramiko
import pyautogui
from annotated_video_parser import VideoParser
import pandas as pd
from shlex import split

ANIMALS = ('aardvark', 'bison', 'canary', 'dalmation', 'emu', 'falcon', 'gnu',
           'hamster', 'impala', 'jellyfish', 'kiwi', 'lemur', 'manatee',
           'nutria', 'okapi', 'porcupine', 'quetzal', 'roadrunner', 'seal',
           'turtle', 'unicorn', 'vole', 'wombat', 'xerus', 'yak', 'zebra')


def get_webrtc_stats(trace, webrtc_dir, browser_type):

    if browser_type != 'chrome':
        return
    
    bname = os.path.basename(trace).split('.')[0]

    json_fname = f'{browser_type}-{bname}-{int(time.time())}.json'
    json_fpath = f'{webrtc_dir}/{json_fname}'

    _ = Popen(f'cp ~/Downloads/webrtc_internals_dump.txt {json_fpath}', shell=True)
    _ = Popen('rm ~/Downloads/webrtc_internals_dump.txt', shell=True)

    return

def get_html5_stats(trace, html5_dir, browser_type):

    if browser_type != 'chrome':
        return

    bname = os.path.basename(trace).split('.')[0]

    json_fname = f'{browser_type}-{bname}-{int(time.time())}.json'
    json_fpath = f'{html5_dir}/{json_fname}'

    _ = Popen(f'cp ~/Downloads/export.json {json_fpath}', shell=True)
    _ = Popen('rm ~/Downloads/export.json', shell=True)

    return

def get_video_quality_stats(rec_dir, rec_start_time):
    fname = [x for x in os.listdir('recordings') if x.endswith('.mp4')][0]
    parser = VideoParser(f'recordings/{fname}')
    fps_values, parser_output = parser.get_fps()
    t = rec_start_time+1
    fps_data = {'fps': fps_values, 'timestamp': [(t + x) for x in range(len(fps_values))]}
    frame_level_data = {'frame_number': [], 'timestamp': []}
    for v in parser_output:
        frame_level_data['frame_number'].append(v[0])
        frame_level_data['timestamp'].append(rec_start_time+(v[1]/1000))
    fps_df = pd.DataFrame(fps_data)
    frame_level_df = pd.DataFrame(frame_level_data)
    fps_df.to_csv(f'{rec_dir}/fps.csv')
    frame_level_df.to_csv(f'{rec_dir}/frame_level_info.csv')
    _ = Popen(f'rm recordings/{fname}', shell=True)

def run_wrapper_script(command):
    # Popen(split(f'ssh root@192.168.1.1 "{command}"'), stdout=PIPE, stderr=PIPE)
    Popen(split(f'ssh root@192.168.1.1 "{command}"'))

def execute(argv=None):
    """ Execute the auto_call CLI command """
    try:
        parser = build_parser()

        args = parser.parse_args(argv)
        bname = os.path.basename(args.trace).split('.')[0]

        if args.capture:
            cap_dir = f'Data/{args.experiment_name}/{bname}/{args.website}/captures'
            make_dir(cap_dir)
            rtc_dir = f'Data/{args.experiment_name}/{bname}/{args.website}/webrtc'
            make_dir(rtc_dir)
            rec_dir = f'Data/{args.experiment_name}/{bname}/{args.website}/rec'
            make_dir(rec_dir)
            html5_dir = f'Data/{args.experiment_name}/{bname}/{args.website}/html5'
            make_dir(html5_dir)
        seperator = '-'

        trace_csv_file = args.trace

        if args.driver:
            print(args.host)

            if args.website == 'meet':
                vca = Meet(args)
            elif args.website == 'zoom':
                vca = Zoom(args)
            elif args.website == 'teams':
                vca = Teams(args)
            elif args.website == 'webex':
                vca = WebEx(args)
            for i in range(args.trials):
                vca.launch_driver()
                if args.browser:

                    vca.launch_app()

                else:
                    vca.launch_client()
                pyautogui.hotkey('f11')
                time.sleep(2)
                # pyautogui.hotkey('ctrl','shift','f10')
                recording_start_time = time.time()
                if vca.host:
                    print('host')
                    time.sleep(200000)
                    quit()
                print("capturing")
                 
                # Capture network traffic or wait necessary time

                print('Shaping Downlink...')
                cmd = f'/root/vcqoe/profiler.sh variable lan1 {args.time} /root/vcqoe/traces/{args.trace}'

                run_wrapper_script(cmd)
                if args.capture: 
                    capture_traffic(args.trace, args, cap_dir)
                else:
                    time.sleep(args.time)
                # pyautogui.hotkey('ctrl','shift','f11')
                time.sleep(2)
                pyautogui.hotkey('f11')
                pyautogui.hotkey('ctrl', 'shift', 'y')
                vca.exit()
                if args.capture:
                    get_webrtc_stats(args.trace, rtc_dir, args.browser)
                    get_html5_stats(args.trace, html5_dir, args.browser)
                    # get_video_quality_stats(rec_dir, recording_start_time)

        if args.metrics:
            if args.website == "zoom":
                metrics = ZoomMetrics()
                metrics.get_past_meeting_instances(args.metrics)

    except Exception as e:
        print('This configuration failed...continuing to next')
        args = parser.parse_args(argv)
        res = Popen('pkill tshark', shell=True)
        res = Popen('pkill chrome', shell=True)
        with open('/home/noise/Projects/vca-qoe-inference/VCA/src/log.txt', 'a') as fd:
            fd.write(f'\nConfig {args.trace[-1]} failed...\n')
            fd.write(e)

def build_parser():
    """ Construct parser to interpret command-line args """

    parser = argparse.ArgumentParser(
        description='Initiate and capture video calls')

    parser.add_argument(
        'website',
        help="Website to visit. Currently supports [zoom] and [meet]",
    )

    parser.add_argument(
        '-d', '--driver',
        default=False,
        action='store_true',
        help='set to launch a call (as opposed to just collecting metrics'
    )

    parser.add_argument(
        '-b', '--browser',
        default=False,
        action='store',
        help='Launch call in browser (as opposed to client)'
    )

    parser.add_argument(
        '-e', '--experiment_name',
        action='store',
        help='Name of experiment (used to name dir with captures)'
    )

    parser.add_argument(
        '-tr', '--trace',
        default=None,
        help='Path to trace CSV file'
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
    return parser


def get_dir(base_name, words=ANIMALS):
    """ Get name of directory for capture to be stored"""

    run_matches = [f for f in pathlib.Path.cwd().glob(f'{base_name}-run-*')]

    next_word = words[len(run_matches)]

    return f'{str(pathlib.Path.cwd())}/{base_name}-run-{next_word}-{int(time.time())}'


def make_dir(new_dir):

    p = pathlib.Path(new_dir)

    if not p.is_dir():
        print('making dir')
        p.mkdir(parents=True)

    return
