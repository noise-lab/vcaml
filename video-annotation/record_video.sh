#!/bin/bash
ffmpeg -video_size 1920x1080 -framerate 60 -f x11grab -i :1 -c:v libx264 -pix_fmt yuv420p $HOME/`date +%H%M%d%m%Y`_output.mp4