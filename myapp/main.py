import sys
from os.path import join, dirname, abspath
import json
import time
import pickle
import random
import argparse
import numpy as np
from functools import partial
from datetime import datetime
import dateutil.tz
import collections

# image and video libraries
from PIL import Image
import imageio
import cv2 
from cv2 import VideoWriter, VideoWriter_fourcc, resize

# app root for flask
APP_ROOT = dirname(abspath(__file__))

# nltk
# if running first time, make sure to uncomment the follwoing two lines
# in later runs commenting improves runtime

# import nltk
# NLTK_DATA_FOLDER = join(APP_ROOT, "static/nltk_data")
# nltk.data.path.append(NLTK_DATA_FOLDER)
# nltk.download('punkt')

from nltk.tokenize import word_tokenize

# warnings
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# typings
from typing import List, Tuple, Any
from nptyping import Array

# utils
from .seq2seq.dataset.prepare_dataset import read_mean_std
from .miscc.utils import *
from .miscc.vis import draw_plate
from .cap_val import CAP_VAL

# config
from .miscc.config import cfg

# torch
import torch
from torch.autograd import Variable

# models - OBJ GAN
from .seq2seq.models import PreEncoderRNN, DecoderRNN

# models - OP GAN
from .model import RNN_ENCODER
from .model import G_NET, CA_NET

# models - super resolution
from .SuperResolution.calculate import evaluate

# super resolution preloading
pb_path = "./SuperResolution/pretrained_model/FALSR-B.pb" # here you can choose between FALSR-A.pb, FALSR-B.pb, FALSR-C.pb

# seeding
manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)


# setting up configurations
embedding_dim = cfg.TEXT.EMBEDDING_DIM
std_img_size = cfg.DATA.IMGSIZE
bs = cfg.TRAIN.BATCH_SIZE[0]
Z_DIM = cfg.GAN.Z_DIM = 100
n_words = 27297  # words in coco caption dataset
net_g = 'op-gan.pth'  # path op gan model
max_objects = 10  # maximum bboxes


# loading coco specific dictionaries to handle category shifts in 2014 dataset
# details: https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/

COCO_JSON_PATH = join(APP_ROOT, "coco_jsons")

with open((join(COCO_JSON_PATH,'coco_ans.json')),'r') as inf:
    coco_annos = eval(inf.read())
    
with open((join(COCO_JSON_PATH,'coco_ans_idx.json')),'r') as inf:
    coco_annos_1_hot = eval(inf.read())
    
with open((join(COCO_JSON_PATH, 'ix2word_label.json')),'r') as inf:
    l_index2word = eval(inf.read())
    
    
m_pth = join(os.path.join(join(APP_ROOT, cfg.DATA.DIR), 'obj_gan'), 'model_objgan.pth')

decoder_path = torch.load(m_pth, map_location=lambda storage, loc: storage)

l_word2ix = {'77': 68,
 '37': 25,
 '22': 50,
 '36': 48,
 '60': 74,
 '61': 15,
 '62': 21,
 '63': 20,
 '64': 32,
 '35': 35,
 '67': 13,
 '82': 18,
 '80': 80,
 '81': 39,
 '86': 24,
 '53': 78,
 '84': 23,
 '85': 45,
 '24': 12,
 '25': 38,
 '23': 69,
 '27': 36,
 '20': 49,
 '21': 61,
 '48': 42,
 '49': 14,
 '46': 44,
 '47': 34,
 '44': 57,
 '32': 66,
 '<sos>': 1,
 '43': 16,
 '40': 62,
 '41': 31,
 '1': 6,
 '3': 7,
 '2': 5,
 '5': 4,
 '4': 17,
 '7': 46,
 '6': 77,
 '9': 70,
 '8': 27,
 '52': 63,
 '28': 47,
 '87': 81,
 '78': 76,
 '42': 72,
 '39': 52,
 '<eos>': 2,
 '65': 59,
 '76': 58,
 '75': 22,
 '74': 67,
 '73': 56,
 '72': 55,
 '70': 60,
 '15': 51,
 '79': 19,
 '14': 71,
 '11': 26,
 '10': 28,
 '13': 41,
 '38': 75,
 '59': 40,
 '58': 11,
 '17': 33,
 '16': 65,
 '19': 43,
 '54': 53,
 '31': 29,
 '56': 54,
 '51': 9,
 '50': 10,
 '<pad>': 0,
 '34': 30,
 '33': 73,
 '55': 82,
 '89': 83,
 '88': 8,
 '18': 64,
 '57': 37,
 '90': 79,
 '<unk>': 3}
    
decoder = DecoderRNN(l_word2index=l_word2ix,
       x_mean = 128.165106,
       y_mean = 135.851983,
       w_mean = 55.95574,
       r_mean = 1.826794,
       batch_size = 1,
       max_len = 150,
       hidden_size = 256,
       gmm_comp_num = 5,
              
       n_layers =1,
       rnn_cell='lstm', bidirectional=True,
    input_dropout_p=0, dropout_p=0.2, use_attention=True
      )
decoder.load_state_dict(decoder_path)

decoder.eval()
decoder.cuda()
    

def load_obj(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
cap_word2index = load_obj(join(os.path.join(join(APP_ROOT, cfg.DATA.DIR), 'obj_gan'), 'cap_word2index.pkl'))
cap_index2word = load_obj(join(os.path.join(join(APP_ROOT, cfg.DATA.DIR), 'obj_gan'), 'cap_index2word.pkl'))


def cap_to_ix(cap: str, cap2ix) -> str:
    """caption to index with coco"""
    return [cap2ix[c] for c in cap.split(' ')]


def coord_converter(coord_seq, mean_x, std_x, mean_y, std_y):
    coord_x_seq, coord_y_seq = [], []
    for i in range(len(coord_seq)):
        x, y = coord_seq[i]
        coord_x_seq.append(x*std_x+mean_x)
        coord_y_seq.append(y*std_y+mean_y)

    return np.array(coord_x_seq), np.array(coord_y_seq)


def validity_indices(x_seq, y_seq, w_seq, h_seq, l_seq):
    x_valid_indices = x_seq > 0
    y_valid_indices = y_seq > 0
    w_valid_indices = w_seq > 0
    h_valid_indices = h_seq > 0
    valid_indices = np.multiply(np.multiply(np.multiply(x_valid_indices, y_valid_indices), w_valid_indices), h_valid_indices)
    x_seq = x_seq[valid_indices]
    y_seq = y_seq[valid_indices]
    w_seq = w_seq[valid_indices]
    h_seq = h_seq[valid_indices]
    l_seq = l_seq[valid_indices]

    return x_seq, y_seq, w_seq, h_seq, l_seq


def ls_one_hot(ls, max_objects=10):
    """transforms ls to one hot format with padding"""
    ls += ['0']*(10 - len(ls))
    ls_1_hot = np.zeros([10,81])
    for i, l in enumerate(ls):
        ls_1_hot[i][coco_annos_1_hot[str(l)]-1] = 1
    return ls_1_hot


def bbox_(xs, ys, ws, hs, max_objects=10):
    """transforms bbox coordinates to a bbox in list format"""
    bbox = 10 * [[-1, -1, -1, -1]]
    for i in range(len(xs)):
        bbox[i] = [xs[i]/256, ys[i]/256, ws[i]/256, hs[i]/256]
    return bbox


def crop_imgs(bbox, max_objects=10):
    """f(bbox) -> bbox_scaled for op-gan generator"""
    ori_size = 268
    imsize = 256

    flip_img = random.random() < 0.5
    img_crop = ori_size - imsize
    h1 = int(np.floor((img_crop) * np.random.random()))
    w1 = int(np.floor((img_crop) * np.random.random()))

    bbox_scaled = np.zeros_like(bbox)
    bbox_scaled[...] = -1.0

    for idx in range(max_objects):
        bbox_tmp = bbox[idx]
        if bbox_tmp[0] == -1:
            break

        x_new = max(bbox_tmp[0] * float(ori_size) - h1, 0) / float(imsize)
        y_new = max(bbox_tmp[1] * float(ori_size) - w1, 0) / float(imsize)

        width_new = min((float(ori_size)/imsize) * bbox_tmp[2], 1.0)
        if x_new + width_new > 0.999:
            width_new = 1.0 - x_new - 0.001

        height_new = min((float(ori_size)/imsize) * bbox_tmp[3], 1.0)
        if y_new + height_new > 0.999:
            height_new = 1.0 - y_new - 0.001

        if flip_img:
            x_new = 1.0-x_new-width_new

        bbox_scaled[idx] = [x_new, y_new, width_new, height_new]
    return bbox_scaled


def get_transformation_matrices(bbox):
    """f(bbox) -> transf. matrix for op-gan generator"""
    bbox = torch.from_numpy(bbox)
    bbox = bbox.view(-1, 4)
    transf_matrices_inv = compute_transformation_matrix_inverse(bbox)
    transf_matrices_inv = transf_matrices_inv.view(max_objects, 2, 3)
    transf_matrices = compute_transformation_matrix(bbox)
    transf_matrices = transf_matrices.view(max_objects, 2, 3)

    return transf_matrices, transf_matrices_inv


def cap_to_bbox(cap: str, text_path: str, box_path: str) -> (Tuple[Array], List[int], Any):
    """
    caption to bbox with labels with OBJ-GAN
    
    cap: caption 
    text_path: path to coco 100 text encoder text_encoder100.pth
    box_path: path to the preloaded box_label folder
    ckpt_path: path to the pretrained model folder
    ckpt_path includes: 
    - cap_index2word.pt
    - cap_word2index.pt
    - label_index2word.pt
    - label_word2index.pt
    - model.pt
    - trainer_states.pt
    
    returns:
    (xs,ys,ws,hs) -> tuple of bounding boxes coordinates like
    ls -> labels list for bounding boxes
    """
    
    # load decoder 
    # decoder = checkpoint.model
    # decoder.eval()
    
    # load gaussian dictionary
    gaussian_dict = np.load(os.path.join(box_path, 'gaussian_dict.npy'), allow_pickle=True).item()
    
    # load text encoder 
    hidden_size = embedding_dim
    encoder = PreEncoderRNN(len(cap_word2index), nhidden=embedding_dim)
    state_dict = torch.load(text_path, map_location=lambda storage, loc: storage)
    encoder.load_state_dict(state_dict)
    encoder.eval()
    
    # use cuda
    if torch.cuda.is_available():
        encoder.cuda()
        
    
    # caption to index
    cap_ix = cap_to_ix(cap, cap_word2index)
    cap_len = [len(cap_ix)]
    # use cuda
    cap_ix = Variable(torch.LongTensor([cap_ix])).cuda()
    
    
    # run the encoder model
    encoder_outputs, encoder_hidden = encoder(cap_ix, cap_len)
    # run the decoder model
    decoder_outputs, xy_gmm_params, wh_gmm_params, decoder_hidden, other = \
    decoder(encoder_hidden, encoder_outputs, is_training=0, early_stop_len=10)
    
    # load mean and std for bbox parameters
    x_mean_std, y_mean_std, w_mean_std, r_mean_std = read_mean_std(os.path.join(box_path, 'mean_std_train2014.txt'))
    
    # convert coordinates for bbox
    xs, ys = coord_converter(other['xy'], x_mean_std[0], x_mean_std[1], y_mean_std[0], y_mean_std[1])
    ws, hs = coord_converter(other['wh'], w_mean_std[0], w_mean_std[1], r_mean_std[0], r_mean_std[1])
    
    # transform bbox parameters
    hs = np.multiply(ws, hs)
    ls = torch.cat(other['sequence']).view(-1,1).transpose(0,1)
    ls = np.array(ls.cpu().data.tolist())[0]
    
    # validity 
    xs, ys, ws, hs, ls = validity_indices(xs, ys, ws, hs, ls)
    
    # eos token and handle labels
    if len(ls) > 1:
        xs = xs[:-1]
        ys = ys[:-1]
        ws = ws[:-1]
        hs = hs[:-1]
        ls = ls[:-1]
        ls = [int(l_index2word[int(l)]) for l in ls]
        ls = np.array(ls)
    
    # filter redundant labels
    counter = collections.Counter(ls)
    unique_labels, label_counts = list(counter.keys()), list(counter.values())
    kept_indices = []
    for label_index in range(len(unique_labels)):
        label = unique_labels[label_index]
        label_num = label_counts[label_index]
        # sample an upper-bound threshold for this label
        mu, sigma = gaussian_dict[label]
        threshold = max(int(np.random.normal(mu, sigma, 1)), 2)
        old_indices = np.where(ls == label)[0].tolist()
        new_indices = old_indices
        if threshold < len(old_indices):
            new_indices = old_indices[:threshold]
        kept_indices += new_indices
    
    kept_indices.sort()
    xs = xs[kept_indices]
    ys = ys[kept_indices]
    ws = ws[kept_indices]
    hs = hs[kept_indices]
    ls = ls[kept_indices]
    ls = [l for l in ls]
    
    xs = xs - ws/2.0
    xs = np.clip(xs, 1, std_img_size-1)
    ys = ys - hs/2.0
    ys = np.clip(ys, 1, std_img_size-1)
    ws = np.minimum(ws, std_img_size-xs)
    hs = np.minimum(hs, std_img_size-ys)
    
    cap_seq = cap_ix[0].tolist()
    
    return (xs, ys, ws, hs), ls


def cap_bbox_to_img(cap: str, bbox_coord: Tuple[Array], ls: List[int], 
               text_path: str, model_path: str, noise = None, local_noise = None, vid=False) -> Array[Image]:
    """
        caption, bbox with labels to img with OP-GAN
        note: One GPU used only
    
    cap: caption 
    bbox_coord: tuple of bounding boxes coordinates like (xs,ys,ws,hs)
    ls: labels list for bounding boxes
    text_path: path to coco 100 text encoder text_encoder100.pth
    model_path: path to model ckpt "op-gan.pth"
    vid: return partial Net G
    
    returns:
    img -> generated image
    """
    # generator network and init weights
    netG = G_NET()
    netG.apply(weights_init)
    
    # generator to cuda
    netG.cuda()
    netG.eval()
    
    # load text encoder
    text_encoder = RNN_ENCODER(n_words, nhidden=embedding_dim)
    state_dict = torch.load(text_path, map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    text_encoder = text_encoder.cuda()
    text_encoder.eval()
    
    # z vector for generator
    if type(noise) == type(None):
        noise = Variable(torch.FloatTensor(bs, Z_DIM)).cuda()
        noise.data.normal_(0, 1)
        
    if type(local_noise) == type(None):
        local_noise = Variable(torch.FloatTensor(bs, 32)).cuda()
        local_noise.data.normal_(0, 1)
    
    # load generator path
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    netG.load_state_dict(state_dict["netG"])
    
    # len caps
    cap_len = Variable(torch.LongTensor([len(cap.split(' '))])).cuda()
    # pad caption
    cap_ix_padded = [cap_word2index[c] 
                     for c in cap.split(' ')][:20] + (20-len([cap_word2index[c] for c in cap.split(' ')])) * [0]
    
    # padded captio to cuda
    cap_ix_padded = Variable(torch.LongTensor([cap_ix_padded])).cuda()
    
    # hidden state text encoder
    hidden = text_encoder.init_hidden(bs)
    
    # encode caption 
    words_embs, sent_emb = text_encoder(cap_ix_padded, cap_len, hidden)
    words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
    mask = (cap_ix_padded == 0)
    num_words = words_embs.size(2)
    if mask.size(1) > num_words:
        mask = mask[:, :num_words]
    
    # labels to one-hot-encoded
    l_1_hot = Variable(torch.FloatTensor([ls_one_hot(ls)])).cuda()
    
    # load bbox coordinates and scale
    bbox = bbox_(*bbox_coord)
    bbox_scaled = crop_imgs(bbox)
    
    # bbox_scaled to transf matrices
    transformation_matrices = get_transformation_matrices(np.single(bbox_scaled))
    
    transf_matrices = transformation_matrices[0].view(bs, 10, 2, 3)
    transf_matrices_inv = transformation_matrices[1].view(bs, 10, 2, 3)
    
    transf_matrices = transf_matrices.cuda()
    transf_matrices_inv = transf_matrices_inv.cuda()
    
    print('sent', sent_emb.size())
    print()
    print('words', words_embs.size())
    if vid:
        netG_partial = partial(netG, sent_emb=sent_emb, word_embs=words_embs, mask=mask, transf_matrices=transf_matrices, transf_matrices_inv=transf_matrices_inv,label_one_hot=l_1_hot, max_objects=10)
                               
        return netG_partial, sent_emb
    
    # generate imgs
    with torch.no_grad():
        fake_imgs, _, c_code = netG(noise, local_noise, sent_emb, words_embs, mask, transf_matrices, transf_matrices_inv, l_1_hot, max_objects)
        
    for j in range(bs):

        k = -1
        # for k in range(len(fake_imgs)):
        im = fake_imgs[k][j].data.cpu().numpy()
        # [-1, 1] --> [0, 255]
        im = (im + 1.0) * 127.5
        im = im.astype(np.uint8)
        im = np.transpose(im, (1, 2, 0))
        im_ = Image.fromarray(im)
        
    return im_, im


def cap_validate(cap: str) -> str:
    """
    validate if the caption fits coco dataset
    if not, it will return a corrected version
    """
    cap_val = CAP_VAL(cap_word2index)
    
    tokens = word_tokenize(cap.lower())
    tokens = ['some' 
              if (w in cap_val.num_word().keys()) or (w in cap_val.num_word().values())
              else w 
              for w in tokens]
    
    words = [word for word in tokens if word.isalpha()]
    
    for i, word in enumerate(words):
            if not word in cap_word2index.keys():
                correct = cap_val.correction(word)
                if not correct in cap_word2index.keys():
                    words[i] = 'person'
                    continue
                words[i] = correct
                
    
    return ' '.join(words)


def cap_to_img(cap: str, text_path: str, box_path: str, 
               obj_path: str, op_path: str, file_path:str) -> Array[Image]:
    """
        caption to img with OBJ-GAN and OP-GAN

    cap: caption 
    text_path: path to coco 100 text encoder text_encoder100.pth
    box_path: path to the preloaded box_label folder
    obj_path: path to the pretrained model folder for OBJ-GAN
    onj_path includes: 
    - cap_index2word.pt
    - cap_word2index.pt
    - label_index2word.pt
    - label_word2index.pt
    - model.pt
    - trainer_states.pt
    op_path: path to model ckpt "op-gan.pth"
    
    returns:
    img -> generated image
    """
    # needs 2.8 seconds to intialize 
    # checkpoint = Checkpoint.load(obj_path)
    
    # check if caption contains problems for coco dataset
    # correct the captio if needed
    cap = cap_validate(cap)
    
    # generate bounding boxes with labels
    bbox, ls = cap_to_bbox(cap, text_path, box_path)

    # draw the bounding boxes with labels
    bbox_drawn= []
    for i in range(bs):
        bbox_drawn.append(draw_plate(np.array([
            *bbox, [ls[i] for i in range(bbox[0].shape[0])]
        ]).transpose()))
    
    # generate the image
    img, _ = cap_bbox_to_img(cap, bbox, ls, text_path, op_path)
    
    # main label for artwork name
    if ls[0] != 0:
        main_ls = coco_annos[str(ls[0])]
    else:
        main_ls = 'random'
    img.save(file_path)
    evaluate([file_path], pb_path=pb_path, save_path="./static/outputs", save=True, scale=2)
    
    # f = file_path.split('/')
    # f_new = '/'.join(f[:-1] + ['SR_512_'+f[-1]])
    # evaluate([f_new], pb_path=pb_path, save_path="./static/outputs", save=True, scale=2)
    return img, bbox_drawn, cap, main_ls


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim=100, n_samples=2):
    # z vector for generator
    noise = np.random.randn(n_samples, latent_dim)
    return noise


def slerp(val, low, high):
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)/1.5
    if so == 0:
        # L'Hopital's rule/LERP
        return (1.0-val) * low + val * high
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high


def interpolate_points(p1, p2, n_steps=30, mode=1):
    # interpolate ratios between the points
    ratios = np.linspace(0, 1, num=n_steps)
    # linear interpolate vectors
    vectors = list()
    for ratio in ratios:
        if mode == 1:
            v = slerp(ratio, p1, p2)
        elif mode == 0:
            v = (1.0 - ratio) * p1 + ratio * p2
        vectors.append(v)
    return np.asarray(vectors)


def make_video(images, outvid=None, fps=30, size=None,
               is_color=True, format="mp4v"):
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
    return vid


def cap_to_vid(cap: str, text_path: str, box_path: str, 
               obj_path: str, op_path: str, file_path_img: str, 
               file_path_vid:str, n_samples: int = 20,
               interpolate: List[str] = ['z','l_z']) -> Array[Image]:
    """
        caption to img with OBJ-GAN and OP-GAN

    cap: caption 
    text_path: path to coco 100 text encoder text_encoder100.pth
    box_path: path to the preloaded box_label folder
    obj_path: path to the pretrained model folder for OBJ-GAN
    onj_path includes: 
    - cap_index2word.pt
    - cap_word2index.pt
    - label_index2word.pt
    - label_word2index.pt
    - model.pt
    - trainer_states.pt
    op_path: path to model ckpt "op-gan.pth"
    
    returns:
    img -> generated image
    """
    # needs 2.8 seconds to intialize 
    # checkpoint = Checkpoint.load(obj_path)
    
    # check if caption contains problems for coco dataset
    # correct the captio if needed
    cap = cap_validate(cap)
    
    # generate bounding boxes with labels
    bbox, ls = cap_to_bbox(cap, text_path, box_path)
    
    # bbox_alt = [np.min((np.max((coor +(random.randint(-1,1)*50)).reshape(-1,1), axis=-1, initial=0)).reshape(-1,1), axis=-1,initial=256) for coor in bbox]
    
    if 'z' in interpolate:
        noise_ = generate_latent_points(n_samples=n_samples)
        noise_ = np.stack((*noise_, noise_[0].copy()), axis=0)
        noise_vec = interpolate_points(*[noise_[0], noise_[1]])
        for n in range(1, n_samples-1):
            noise_vec = np.concatenate((noise_vec, interpolate_points(*[noise_[n], noise_[n+1]])[1:]), axis=0)
        noise_vec = np.concatenate((noise_vec, interpolate_points(*[noise_[-2], noise_[-1]])[1:-1]), axis=0)
        noise_vec = torch.FloatTensor(noise_vec).cuda()   
    else: 
        noise_vec = None
         
    
    if 'l_z' in interpolate:
        local_noise_ = generate_latent_points(latent_dim=32, n_samples=n_samples)
        local_noise_ = np.stack((*local_noise_, local_noise_[0].copy()), axis=0)
        local_noise_vec = interpolate_points(*[local_noise_[0], local_noise_[1]])
        for n in range(1, n_samples-1):
            local_noise_vec = np.concatenate((local_noise_vec, interpolate_points(*[local_noise_[n], local_noise_[n+1]])[1:]), axis=0)
        local_noise_vec = np.concatenate((local_noise_vec, interpolate_points(*[local_noise_[-2], local_noise_[-1]])[1:-1]), axis=0)
        local_noise_vec = torch.FloatTensor(local_noise_vec).cuda()
    else: 
        local_noise_vec = None
    
    imgs = []
    imgs_pill = []
    
    netG_partial, sent_emb = cap_bbox_to_img(cap, bbox, ls, text_path, op_path, vid=True)
    
    # first image sets the c_code vector for the rest
    fake_imgs, _, c_code = netG_partial(z_code=noise_vec[0].view(1,-1), local_noise=local_noise_vec[0].view(1,-1))
    
    im = fake_imgs[-1][0].data.cpu().numpy()
    # [-1, 1] --> [0, 255]
    im = (im + 1.0) * 127.5
    im = im.astype(np.uint8)
    im = np.transpose(im, (1, 2, 0))
    im_ = Image.fromarray(im)
    imgs.append(im)
    
    Image.fromarray(im).save(file_path_img)
    
    # super res for single image in flask app
    # evaluate([file_path_img], pb_path=pb_path, save_path="./static/outputs", save=True, scale=2)
    
    # f = file_path_img.split('/')
    # img_pill_512 = '/'.join(f[:-1] + ['SR_512_'+f[-1]])
                            
    
    # evaluate([img_pill_512], pb_path=pb_path, save_path="./static/outputs", save=True, scale=2)
    
    
    i = 0
    print(c_code)
    
    c_vec = np.array([c_code.detach().cpu().numpy()])
    
    for n in range(0, n_samples-1):
        
        c_vec = np.concatenate((c_vec, interpolate_points(*[c_vec[-1], c_code.detach().cpu().numpy()+2*np.random.randn(1,100)], mode=0)[1:]), axis=0)
    c_vec = np.concatenate((c_vec, interpolate_points(*[c_vec[-1], c_code.detach().cpu().numpy()], mode=0)[1:-1]), axis=0)
    
    c_vec = torch.FloatTensor(c_vec).cuda()
    # c_add_ = [torch.zeros([1,100]).cuda()] + [torch.ones([1,100]).cuda() for i in range(n_samples-1)] + [torch.zeros([1,100]).cuda()]
    
    print(c_vec.size())
    
    # generate the images
    for noise,local_noise,c in zip(noise_vec[1:], local_noise_vec[1:], c_vec[1:]):
        fake_imgs, _, _ = netG_partial(z_code=noise.view(1,-1), local_noise=local_noise.view(1,-1), c_code=c_code)
        im = fake_imgs[-1][0].data.cpu().numpy()
        # [-1, 1] --> [0, 255]
        im = (im + 1.0) * 127.5
        im = im.astype(np.uint8)
        im = np.transpose(im, (1, 2, 0))
        imgs.append(im)
        # Image.fromarray(im).save('./superres/'+'image_'+str(i)+'.png')
        # imgs_pill.append('./superres/'+'image_'+str(i)+'.png')
        i = i+1
    
       
    # evaluate(imgs_pill, pb_path=pb_path, save_path="./superres", save=True, scale=2)
    
    # f = [p.split('/') for p in imgs_pill]
    # imgs_pill_512 = ['/'.join(p[:-1] + ['SR_512_'+p[-1]]) for p in f]
    # f_ = [p.split('/') for p in imgs_pill_512]
    # imgs_pill_1024 = ['/'.join(p[:-1] + ['SR_1024_'+p[-1]]) for p in f_]
        
    # for super resolution uncommend below
    # evaluate(imgs_pill_512, pb_path=pb_path, save_path="./superres", save=True, scale=2)
    
    video = make_video(imgs, file_path_vid)
    # imgs_1024 = []
    # for img_ in imgs_pill_1024:
    #     imgs_1024.append(imageio.imread(img_))
    
    # imgs_512 = []
    # for img_ in imgs_pill_512:
    #     imgs_512.append(imageio.imread(img_))
    # 
    # timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S').replace(" ", "_").replace(':', '_').replace('-', '_')
    # title = 'video_{0}_{1}.mp4'.format(cap.replace(" ", "_"), timestamp)
    # video = make_video(imgs_512, file_path_vid)
    
    
    # main label for artwork name
    if ls[0] != 0:
        main_ls = coco_annos[str(ls[0])]
    else:
        main_ls = 'random'
    
    return video, main_ls


# loading partial configs to the cap_to_img function
# change the op_path directory if needed
cap_img_partial = partial(
    cap_to_img, 
    text_path = os.path.join(join(APP_ROOT, cfg.DATA.DIR), cfg.DATA.PRET, cfg.DATA.ENC), 
    box_path = os.path.join(APP_ROOT, cfg.DATA.DIR, 'box_label'), 
    obj_path = os.path.join(join(APP_ROOT, cfg.DATA.DIR), cfg.DATA.PRET),
    op_path = join(APP_ROOT, cfg.DATA.DIR, cfg.DATA.PRET, "op-gan.pth")
)


# loading partial configs to the cap_to_img function
# change the op_path directory if needed
cap_vid_partial = partial(
    cap_to_vid, 
    text_path = os.path.join(join(APP_ROOT, cfg.DATA.DIR), cfg.DATA.PRET, cfg.DATA.ENC), 
    box_path = os.path.join(APP_ROOT, cfg.DATA.DIR, 'box_label'), 
    obj_path = os.path.join(join(APP_ROOT, cfg.DATA.DIR), cfg.DATA.PRET),
    op_path = join(APP_ROOT, cfg.DATA.DIR, cfg.DATA.PRET, "op-gan.pth")
)


#if __name__ == "__main__":
    # img is PIL Image from array
    #imgs, bbox, caption, _ = cap_img_partial('an apple on a table')
    #print(imgs)

#     parser = argparse.ArgumentParser(description='Train a AttnGAN network')
#     parser.add_argument('--cap', dest='cap', help='caption', type=str,
#                         default='some pizzas on a table')
#     parser.add_argument('--output_dir', dest='output_dir', help='output directory',
#                         type=str, default='./outputs')
#     args = parser.parse_args()
#
#     if not os.path.isdir(args.output_dir):
#         os.mkdir(args.output_dir)
#         print('created output directory at {}'.format(args.output_dir))
#
#     start = time.time()
#     imgs, bbox_drawn, cap, main_ls = cap_img_partial(args.cap)
#
#     print('generated bboxes and image from caption in {: .2f} seconds'.format(time.time() - start))
#
#     now = datetime.datetime.now(dateutil.tz.tzlocal())
#     timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
#
#     imgs[0].save(os.path.join(args.output_dir, 'image_{0}_{1}.jpg'.format(main_ls, timestamp)))
#     bbox_drawn[0].save(os.path.join(args.output_dir, 'bbbox_{0}_{1}.jpg'.format(main_ls, timestamp)))
#     text_file = open(os.path.join(args.output_dir, 'cap_{0}_{1}.txt'.format(main_ls, timestamp)), "w")
#     text_file.write(args.cap + ' --> ' + cap)
#     text_file.close()
#
#     print('saved image, bbox an caption to {}'.format(args.output_dir))
