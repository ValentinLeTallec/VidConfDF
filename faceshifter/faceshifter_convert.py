# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/1jyNj69Ac-rNJbIDoPTOBKsGmUveVpGRV
"""


"""# Clone code and download weights"""
# clone model
# !git clone https://github.com/richarduuz/Research_Project

# Download weights
# cd /content/Research_Project/ModelC/face_modules/
# gdown https://drive.google.com/uc?id=15nZSJ2bAT3m-iCBqP3N_9gld5_EGv4kp
# cd /content/Research_Project/ModelC/saved_models/
# gdown https://drive.google.com/uc?id=1iANX7oJoXCEECNzBEW1xOpac2tDOKeu9

"""# Load Model"""

import sys
from os.path import (join, isfile, basename, dirname, abspath)
from os import listdir, makedirs, getcwd, chdir
root = dirname(abspath(__file__))
work_dir = getcwd()
chdir(root)
sys.path.append(join(root, "face_modules"))

from tqdm import tqdm
import re
import argparse
import numpy as np
import PIL.Image as Image
import cv2
from face_modules.mtcnn import *
from network.AEI_Net import *
from face_modules.model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
import torch.nn.functional as F             # pylint: disable=import-error
import torchvision.transforms as transforms # pylint: disable=import-error
import torch                                # pylint: disable=import-error
from datetime import datetime


detector = MTCNN()
device = torch.device('cuda')
G = AEI_Net(c_id=512)
G.eval()
G.load_state_dict(torch.load('./saved_models/G_latest.pth',
                  map_location=torch.device('cpu')))
G = G.cuda()

arcface = Backbone(50, 0.6, 'ir_se').to(device)
arcface.eval()
arcface.load_state_dict(torch.load(
    './face_modules/model_ir_se50.pth', map_location=device), strict=False)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
chdir(work_dir)

"""## Inference"""


def faceShifterTmp(i1, i2):
    donor_img_name = "./{}.jpg".format(i1)
    receiver_img_name = "./{}.jpg".format(i2)
    faceShifter(receiver_img_name, donor_img_name)


def path2id(string: str):
    # Image_name = f"{identity:0>3}/{video_name}_{frame_nb:0>5}_{identity:0>3}.jpg"
    #       OR if their is more than one image that would have had this name :
    # Image_name = f"{identity:0>3}/{video_name}_{frame_nb:0>5}_{identity:0>3}_{j:0>3}.jpg"
    # Except that if the images were then sorted manually then the `identity` in the file name might be wrong
    # id = f"{video_name}_{identity:0>3}"

    # string = re.sub(r"\.[^\.]*$", "", string) # Remove extension
    # video_name = re.search(r"/[^/]*@\d\d:\d\d:\d\d", string).group()
    # string_end = re.sub(r"^[^/]*@\d\d:\d\d:\d\d(_extended)?_", "", string)
    # identity = string_end.split("_")[1]
    # string = /some/random/path/pt-wO73Y8to@00:09:15/manual/000_faces/pt-wO73Y8to@00:09:15_00156_000_0.png
    try:
        if not (re.search(r"/[^/]*@\d\d:\d\d:\d\d", string) is None):
            video_name = re.search(r"/[^/]*@\d\d:\d\d:\d\d", string).group()[1:]

            identity = re.search(r"/[^/]*@\d\d:\d\d:\d\d(/manual)?/\d\d\d/[^/]*@\d\d:\d\d:\d\d", string).group()
            identity = re.sub(r"^/[^/]*@\d\d:\d\d:\d\d(/manual)?/", "", identity)
            identity = re.sub(r"/[^/]*@\d\d:\d\d:\d\d$", "", identity)
        elif not (re.search(r"/laptop_a\d{4}_(in|out)door", string) is None):
            video_name = re.search(r"/laptop_a\d{4}_(in|out)door", string).group()[1:]

            identity = re.search(r"/laptop_a\d{4}_(in|out)door(/manual)?/\d\d\d/laptop_a\d{4}_(in|out)door", string).group()
            identity = re.sub(r"^/laptop_a\d{4}_(in|out)door(/manual)?/", "", identity)
            identity = re.sub(r"/laptop_a\d{4}_(in|out)door$", "", identity)
    except Exception as e:
        print(string)
        raise(Exception(e))
    return video_name + '_' + identity


def faceShifter(receiver_img_name: str, donor_img_name: str,
                out_img_folder='./', out_img_name='{receiver_file}⟶{donor_file}.jpg', throwError = True):

    donor_img_name    = abspath(join(work_dir, donor_img_name))
    receiver_img_name = abspath(join(work_dir, receiver_img_name))
    out_img_folder    = abspath(join(work_dir, out_img_folder))

    Xd_raw = cv2.imread(donor_img_name)
    if Xd_raw is None:
        raise FileNotFoundError("\tDonor image not found : \n" + donor_img_name)
    try:
        Xd = detector.align(Image.fromarray(
            Xd_raw[:, :, ::-1]), crop_size=(256, 256))
    except Exception as e:
        if throwError :
            print(e)
            print('{} (Donor) could not be aligned'.format(donor_img_name))
        with open(join(root, "errDonor.txt"), 'a') as f: f.write('[{}]  {}\n'.format(str(datetime.now())[:16], donor_img_name))
        return 'errDonor'

    if Xd is None:
        return 'errDonor'

    Xd_raw = np.array(Xd)[:, :, ::-1]
    Xd = test_transform(Xd)
    Xd = Xd.unsqueeze(0).cuda()

    with torch.no_grad():
        embeds = arcface(F.interpolate(
            Xd[:, :, 19:237, 19:237], (112, 112), mode='bilinear', align_corners=True))

    Xr_raw = cv2.imread(receiver_img_name)
    if Xr_raw is None:
        raise FileNotFoundError("\tReceiver image not found : \n" + receiver_img_name)

    try:
        Xr, trans_inv = detector.align(Image.fromarray(
            Xr_raw[:, :, ::-1]), crop_size=(256, 256), return_trans_inv=True)
    except Exception as e:
        if throwError :
            print(e)
            print('{} (Receiver) could not be aligned'.format(receiver_img_name))
        with open(join(root, "errReceiver.txt"), 'a') as f: f.write('[{}]  {}\n'.format(str(datetime.now())[:16], receiver_img_name))
        return 'errReceiver'

    if Xr is None:
        return 'errReceiver'

    Xr_raw = Xr_raw.astype(np.float)/255.0
    Xr = test_transform(Xr)
    Xr = Xr.unsqueeze(0).cuda()

    mask = np.zeros([256, 256], dtype=np.float)
    for i in range(256):
        for j in range(256):
            dist = np.sqrt((i-128)**2 + (j-128)**2)/128
            dist = np.minimum(dist, 1)
            mask[i, j] = 1-dist
    mask = cv2.dilate(mask, None, iterations=20)

    with torch.no_grad():
        Yr, _ = G(Xr, embeds)
        Yr = Yr.squeeze().detach().cpu().numpy().transpose([1, 2, 0])*0.5 + 0.5
        Yr = Yr[:, :, ::-1]
        Yr_trans_inv = cv2.warpAffine(Yr, trans_inv, (np.size(
            Xr_raw, 1), np.size(Xr_raw, 0)), borderValue=(0, 0, 0))
        mask_ = cv2.warpAffine(mask, trans_inv, (np.size(
            Xr_raw, 1), np.size(Xr_raw, 0)), borderValue=(0, 0, 0))
        mask_ = np.expand_dims(mask_, 2)
        Yr_trans_inv = mask_*Yr_trans_inv + (1-mask_)*Xr_raw

        donor_img_basename = basename(donor_img_name)
        receiver_img_basename = basename(receiver_img_name)

        out_img_folder = out_img_folder.format(donor_file    = donor_img_basename,
                                               receiver_file = receiver_img_basename,
                                               #donor    = path2id(donor_img_name),
                                               #receiver = path2id(receiver_img_name)
                                               )

        out_img_name = out_img_name.format(donor_file    = donor_img_basename,
                                           receiver_file = receiver_img_basename,
                                           #donor    = path2id(donor_img_name),
                                           #receiver = path2id(receiver_img_name)
                                           )

        makedirs(out_img_folder, exist_ok=True)
        cv2.imwrite(join(out_img_folder, out_img_name), Yr_trans_inv*255)



if __name__ == "__main__":

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="An photo of the donnor")
    ap.add_argument("-f", "--folder", required=True,
                    help="Folder containing the receiver photos")
    ap.add_argument("-o", "--output", type=str,
                    default="datasets/VideoConference/VideoConference-synthesis/images/faceshifter/",
                    help="path to the output directory")
    args = vars(ap.parse_args())

    donor_img = args["image"]
    receiver_dir = args["folder"]
    out_path = args["output"]

    receiver_faces_names = [join(receiver_dir, f) for f in listdir(receiver_dir) if isfile(join(receiver_dir, f))]
    for receiver in tqdm(receiver_faces_names):
        faceShifter(receiver, donor_img, join(out_path, '{receiver}⟶{donor}'), '{receiver_file}')

# py faceshifter/faceshifter_convert.py -i datasets/VideoConference/VideoConference-real/images/__ -f datasets/VideoConference/VideoConference-real/images/__
