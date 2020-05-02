import cv2
from PIL import Image

from datetime import datetime

import imageio


from cv2 import VideoWriter, VideoWriter_fourcc, resize

from os.path import join, dirname, abspath
import os

from SuperResolution.calculate import evaluate

# # super resolution preloading
pb_path = "./SuperResolution/pretrained_model/FALSR-B.pb" # here you can choose between FALSR-A.pb, FALSR-B.pb, FALSR-C.pb
# 
imgs_pill = []
# 
vidcap = cv2.VideoCapture('./out_video/video_very_many_sheeps_on_the_field_near_some_houses_2020_03_18_12_14_12.mp4')
success,image = vidcap.read()
count = 0
while success:
    imgs_pill.append("./out_video/imgs/frame%d.png" % count)
    cv2.imwrite("./out_video/imgs/frame%d.png" % count, image)     # save frame as JPEG file      
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
    

evaluate(imgs_pill, pb_path=pb_path, save_path="./out_video/imgs", save=True, scale=2)

f = [p.split('/') for p in imgs_pill]
imgs_pill_512 = ['/'.join(p[:-1] + ['SR_512_'+p[-1]]) for p in f]
f_ = [p.split('/') for p in imgs_pill_512]
imgs_pill_1024 = ['/'.join(p[:-1] + ['SR_1024_'+p[-1]]) for p in f_]
# f__ = [p.split('/') for p in imgs_pill_1024]
# imgs_pill_2048 = ['/'.join(p[:-1] + ['SR_2048_'+p[-1]]) for p in f__]


evaluate(imgs_pill_512, pb_path=pb_path, save_path="./out_video/imgs", save=True, scale=2)
# evaluate(imgs_pill_1024, pb_path=pb_path, save_path="./out_video/imgs", save=True, scale=2)
# 

imgs_1024 = []
for img_ in imgs_pill_1024:
    imgs_1024.append(imageio.imread(img_, ))
    
# imgs_1024 = []
# for img_ in imgs_pill_2048:
#     imgs_1024.append(imageio.imread(img_, ))
    
print(len(imgs_1024))

def make_video(images, outvid=None, fps=30, size=None,
               is_color=True, format="x264"):
    # install on ubuntu:
    # sudo apt-get install ffmpeg x264 libx264-dev
    """
    Create a video from a list of images.
 
    @param      outvid      output video
    @param      images      list of images to use in the video
    @param      fps         frame per second
    @param      size        size of each frame
    @param      is_color    color
    @param      format      see http://www.fourcc.org/codecs.php
    @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
 
    The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
    By default, the video will have the size of the first image.
    It will resize every image to this size before adding them to the video.
    """
    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for img in images:
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        img_ = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        vid.write(img_)
    vid.release()
    cv2.destroyAllWindows()
    return vid
    
timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S').replace(" ", "_").replace(':', '_').replace('-', '_')
title = 'video_{0}_{1}.mp4'.format('video_very_many_sheeps_on_the_field_near_some_houses_2020_03_18_12_14_12', timestamp)
video = make_video(imgs_1024, './superres_1024/'+title)
