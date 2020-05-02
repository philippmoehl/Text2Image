from glob import glob
import pandas as pd
import sys
import ntpath
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from nltk.tokenize import RegexpTokenizer
from skimage.transform import resize
import os
import errno

checkpoint = '2018_10_23_14_55_12'
gen_dir = '../data/coco/gen_masks_%s/*'%(checkpoint)
gt_dir = '../data/coco/masks/'
img_dir = '../data/coco/images/'
txt_dir = '../data/coco/text/'
# output_dir = '../vis_bboxes_%s/'%(checkpoint)
CAPS_PER_IMG = 5
FONT_MAX = 40
FONT_REAL = 30
MAX_WORD_NUM = 20
#FNT = ImageFont.truetype('../data/coco/share/Pillow/Tests/fonts/FreeMono.ttf', FONT_REAL)
STD_IMG_SIZE = 256
VIS_SIZE = STD_IMG_SIZE
OFFSET = 2
SHOW_LIMIT = 500

def path_leaf(path):
	return ntpath.basename(path)

def load_captions(cap_path):
    all_captions = []
    with open(cap_path, "r") as f:
        captions = f.read().decode('utf8').split('\n')
        cnt = 0
        for cap in captions:
            if len(cap) == 0:
                continue
            cap = cap.replace("\ufffd\ufffd", " ")
            # picks out sequences of alphanumeric characters as tokens
            # and drops everything else
            tokenizer = RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(cap.lower())
            # print('tokens', tokens)
            if len(tokens) == 0:
                print('cap', cap)
                continue

            tokens_new = []
            for t in tokens:
                t = t.encode('ascii', 'ignore').decode('ascii')
                if len(t) > 0:
                    tokens_new.append(t)
            sentence = ' '.join(tokens_new)
            all_captions.append(sentence)
            cnt += 1
            if cnt == CAPS_PER_IMG:
                break
        if cnt < CAPS_PER_IMG:
            print('ERROR: the captions for %s less than %d'
                  % (cap_path, cnt))
    return all_captions

def draw_plate(bboxes):
	bbox_plate = Image.fromarray((np.ones((VIS_SIZE, VIS_SIZE, 3))*255).astype(np.uint8))
	if bboxes is None:
		return bbox_plate

	d = ImageDraw.Draw(bbox_plate)
	for i in range(bboxes.shape[0]):
		left, top, width, height, label = bboxes[i, :5]
		label = int(label)
		color = (210-label*2,label*3,50+label*2)
		d.rectangle([left, top, left+width-1, top+height-1], outline=color)
		d.text([left+5, top+5], str(label), fill=color)
	del d
	return bbox_plate

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def is_non_zero_file(fpath):  
    return True if os.path.isfile(fpath) and os.path.getsize(fpath) > 0 else False

