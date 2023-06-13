from qr_processor import QRProcessor
import cv2
import numpy as np

class VideoParser:

    def __init__(self, video_file):
        self.video_file = video_file
        self.fps = 60

    def parse(self): # No need to crop the video frame. Works as is.
        cap = cv2.VideoCapture(self.video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        values = []
        qr = QRProcessor()
        prev_ts = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if frame is not None:
                num = qr.decode(frame)
                rel_ts = cap.get(cv2.CAP_PROP_POS_MSEC)
                if rel_ts == 0:
                    curr_ts = prev_ts + 1000/fps
                else:
                    curr_ts = rel_ts
                values.append((num-1, curr_ts))
                prev_ts = curr_ts
            else:
                break
        cap.release()
        return values

    def get_fps(self):
        print('Parsing video...')
        vals = self.parse()
        frame_set = set(vals)
        p = vals[0][0]
        i = 0
        count = 0
        missing = []
        res_fps = []
        next_ts = 1
        frames = set()
        print('Calculating FPS...')
        while i < len(vals):
            if vals[i][0] < 0:
                if vals[i][1]/1000 > next_ts:
                    res_fps.append(len(frames))
                    frames = set()
                    next_ts += 1
                i += 1
                continue
            frames.add(vals[i][0])
            if vals[i][0] == p:
                p = vals[i][0]
                if vals[i][1]/1000 > next_ts:
                    res_fps.append(len(frames))
                    frames = set()
                    next_ts += 1
                i += 1
                continue
            if vals[i][0] != (p+1)%500:
                k = (p+1)%500
                while k != vals[i][0]:
                    print(f'i = {i}, k = {k}')
                    missing.append(k)
                    k = (k+1)%500
                    count += 1
            p = vals[i][0]
            if vals[i][1]/1000 > next_ts:
                res_fps.append(len(frames))
                frames = set()
                next_ts += 1
            i += 1
        print('Count = ', count)
        print('Missing frames = ', missing)
        return res_fps

if __name__ == '__main__':
<<<<<<< HEAD
    vals = VideoParser('edited_video.mp4').parse()
    print(len(set(vals)))
=======
    fps = VideoParser('test_video.mp4').get_fps()
    print(fps)
>>>>>>> e9397995cad4c080cbef0576eb57a1fb8ec162fa
