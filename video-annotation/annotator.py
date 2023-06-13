from qr_processor import QRProcessor
import cv2
import numpy as np

class VideoAnnotator:
    def __init__(self, video_file):
        self.video_file = video_file

    def annotate(self):
        cap = cv2.VideoCapture(self.video_file)
        idx = 1
        qr = QRProcessor()
        edited_frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if frame is not None:
                print(f'Processing frame {idx}...')
                h, w, layers = frame.shape
                size = (w, h)
                qr.generate(idx, f'frame_qrcodes/frame_{idx}.png')
                qr_image = cv2.imread(f'frame_qrcodes/frame_{idx}.png')
                frame[500:500+qr_image.shape[0], 500:500+qr_image.shape[1]] = qr_image
                edited_frames.append(frame)
                idx += 1
            else:
                break
        print('Writing edited images to file...')
        out = cv2.VideoWriter('edited_video.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 25, size)
        for i in range(len(edited_frames)):
            print(f'Writing frame {i}...')
            out.write(edited_frames[i])
        cap.release()
        out.release()

if __name__ == '__main__':
    VideoAnnotator('video.mp4').annotate()