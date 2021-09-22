from os.path import join, exists, basename
from sqlite3.dbapi2 import Cursor
from six.moves import cPickle as pickle #for performance
from shutil import move
from os import makedirs, listdir, rmdir
import sys, re
from typing import List
import cv2
import sqlite3
import warnings
from numpy.core.numeric import Inf
from tqdm import tqdm
import numpy as np
import argparse
from face_recognition import face_encodings

PREFIX_RETINA = 'models/retinaface-R50/R50'
RETINA_FOLDER = './insightface/detection/RetinaFace/'
sys.path.insert(1, RETINA_FOLDER)
import random
verbose = False
# HCVM stand for "Human centric video matting challenge" (https://maadaa.ai/cvpr2021-human-centric-video-matting-challenge/)

from face_detect import (RetinaFace, get_videos_infos, save_face,
                         save_log, get_annotations, clean_annotations, 
                         save_face_tmp, parse_string, clusterFaces)

def detect_crop_and_identify(dataset, method, compression, label, nb_frames_processed,
                             random_sample, db_folder, db_file, output_folder, face_size_threshold = None):

    detector = RetinaFace(PREFIX_RETINA, 0, 0, network='net3', vote=True)
    
    makedirs(db_folder, exist_ok=True)
    print("Saved in database : ", join(db_folder, db_file))
    conn = sqlite3.connect(db_folder + db_file)
    db = conn.cursor()

    video_files = get_videos_infos(db, dataset, method, compression, label)
    video_files = list(filter(lambda e: not (re.match(r"^laptop_a\d{4}_(in|out)door\.mp4$", e[3]) is None),
                              video_files)) # Only keep HCVM files (cf. face_detect_hcvm.py)
    print("Processed {} videos".format(len(video_files)))

    annotations_all = []
    for (videoid, dataset_folder, video_folder, video_file, nb_frames,
         frames) in tqdm(video_files):

        # Select random frames
        if frames == None:
            available_frames = range(nb_frames)
        else:
            available_frames = list(
                set(range(nb_frames)) -
                set(map(lambda x: int(x), frames.split(','))))
        nb_frames_processed = min(nb_frames_processed, len(available_frames))
        if random_sample:
            subset_frame = random.sample(available_frames, nb_frames_processed)
        else:
            subset_frame = available_frames[:nb_frames_processed]

        # Load video
        video_path = db_folder + dataset_folder + '/' + video_folder + '/' + video_file
        video = cv2.VideoCapture(video_path)

        # Create output folder
        video_name = re.sub(r"\.[^\.]*$", "", video_file)
        # images_folder = join(dataset_folder, re.sub("/videos$", "",  video_folder), 'images', video_name)
        images_folder = join(dataset_folder, re.sub("/videos$", "",  video_folder), "hcvm")
        tmp_folder = output_folder + images_folder + "/tmp/"
        # if the folder already exist, we concidere the job has already be done
        # before and we skip to next iteration
        # if exists(join(output_folder, images_folder)) : continue

        makedirs(join(output_folder, images_folder), exist_ok=True)

        # Save frames
        for frame_nb in tqdm(subset_frame):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_nb)
            ret, frame = video.read()
            if not ret:
                log_message = 'file ' + video_file + ' stopped at frame ' + str(
                    frame_nb) + ' (max ' + str(nb_frames) + ')'
                save_log('log', [log_message], prefix='Error: ')
                break

            annotations = get_annotations(frame, detector)
            annotations, bad_detection = clean_annotations(annotations, face_size_threshold)
            
            db.execute("INSERT INTO frames(videoid,frame_nb) VALUES (?,?)",
                       (videoid, frame_nb))
            frameid = db.execute("SELECT last_insert_rowid()").fetchone()[0]
            if bad_detection:
                infos = [dataset_folder, video_path, str(annotations), '']
                save_log(output_folder + 'error_detect.txt',
                         infos,
                         prefix='Error: ')

            i = 0
            for annotation in annotations:
                image_path_tmp = tmp_folder + "{}_{:0>5}_{:0>3}.jpg".format(video_name, frame_nb, i)

                face, x, y, w, h, p = save_face_tmp(frame, image_path_tmp, annotation)
                i += 1

                face_boxes = [(0,face.shape[1],face.shape[0],0)]

                annotations_all.append({"image_path_tmp": image_path_tmp,
                                        "annotation" : annotation, 
                                        "crop_result": (x, y, w, h, p),
                                        "frame_nb"   : frame_nb,
                                        "frameid"    : frameid,
                                        "video_name" : video_name,
                                        "embedding"  : face_encodings(face, face_boxes)[0]})
        conn.commit()

    
    makedirs(join(output_folder, images_folder), exist_ok=True)

    # Cluster Identities
    with open(join(output_folder, images_folder, "annotations_all.pkl"), 'wb') as f: pickle.dump(annotations_all, f)
    # with open(join(output_folder, images_folder, "annotations_all.pkl"), 'rb') as f: annotations_all = pickle.load(f)
    if len(annotations_all) > 0:
        annotations_all = clusterFaces(annotations_all)
    else:
        warnings.warn("No faces were detected")

    image_names = []
    for i in range(len(annotations_all)):
        annotation  = annotations_all[i]["annotation"]
        frame_nb    = annotations_all[i]["frame_nb"]
        frameid     = annotations_all[i]["frameid"]
        identity    = annotations_all[i]["identity"]
        video_name  = annotations_all[i]["video_name"]
        crop_result    = annotations_all[i]["crop_result"]
        image_path_tmp = annotations_all[i]["image_path_tmp"]

        identity_folder = join(images_folder, "{:03}".format(identity))

        image_name = "{}_{:0>5}_{:0>3}.jpg".format(video_name, frame_nb, identity)

        # Manage de possibility that multiple face of the same frame
        # are concidered as having the same identity
        # (while the name already exist : create new name)
        j = 0
        while image_name in image_names:
            image_name = "{}_{:0>5}_{:0>3}_{:0>3}.jpg".format(video_name, frame_nb, identity,j)
            j += 1
        image_names.append(image_name)

        annotations_all[i]["image_path"] = join(output_folder, identity_folder, image_name)

        save_face(db, crop_result, identity_folder, image_name, identity,
                  frameid, image_path_tmp, output_folder)                       

        conn.commit()
        if len(listdir(tmp_folder)) < 1:
            rmdir(tmp_folder)
    with open(join(output_folder, images_folder, "annotations_all.pkl"), 'wb') as f: pickle.dump(annotations_all, f)
    conn.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--db",
                        default='./data/',
                        help="Path to the database")
    parser.add_argument("--dbFile",
                        default="Deepfake.db",
                        help="Path to the database (.db file)")
    parser.add_argument("-O",
                        "--output",
                        default="None",
                        type=str,
                        help="Where should the output be stored")
    parser.add_argument("-d",
                        "--dataset",
                        default="None",
                        type=str,
                        help="Which dataset to be considered")
    parser.add_argument("-m",
                        "--method",
                        default="None",
                        type=str,
                        help="Which method to be considered")
    parser.add_argument("-l",
                        "--label",
                        default="REAL",
                        type=str,
                        help="Which label to be considered")
    parser.add_argument("-c",
                        "--compression",
                        default="None",
                        type=str,
                        help="Which compression to be considered")
    parser.add_argument("-n",
                        "--nbFrames",
                        default=Inf,
                        type=int,
                        help="Number of frames per video to be considered")
    parser.add_argument("-r",
                        "--random",
                        default=False,
                        type=bool,
                        help="Select random frames or in natural order")
    parser.add_argument("-s",
                        "--face_size_threshold",
                        default=0.5,
                        type=float,
                        help="Faces are ignored : \
                            if (s >  1 and 'face width' < s) or \
                               (s <= 1 and 'face area'  < 'area of the biggest face' * s),\
                            recommended values s=0.5 or s=40")
    parser.add_argument("-v", "--verbose",
                        action='store_true',
                        help="Print more info")
    args = parser.parse_args()
    DB_FOLDER = parse_string(args.db, isFolder=True)
    DB_FILE = parse_string(args.dbFile)
    OUTPUT_FOLDER = DB_FOLDER if args.output == "None" else parse_string(
        args.output, isFolder=True)
    DATASET = parse_string(args.dataset)
    METHOD = parse_string(args.method)
    LABEL = parse_string(args.label)
    COMPRESSION = parse_string(args.compression)
    NB_FRAMES = args.nbFrames
    RANDOM = args.random
    FACE_SIZE_THRESHOLD = args.face_size_threshold

    verbose = args.verbose
    
    detect_crop_and_identify(DATASET, METHOD, COMPRESSION, LABEL, NB_FRAMES, RANDOM,
                             DB_FOLDER, DB_FILE, OUTPUT_FOLDER, FACE_SIZE_THRESHOLD)

    # The script is assumed to be run with docker :
    # docker run --gpus all -v "path/to/datasets/:/app/data" 
