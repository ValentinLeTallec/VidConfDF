from os.path import join, exists, basename
from glob import glob
from sqlite3.dbapi2 import Cursor
from six.moves import cPickle as pickle #for performance
from shutil import move
from os import makedirs, listdir, rmdir
import sys, re
from typing import List
import cv2
import sqlite3
import warnings
from datetime import datetime
from numpy.core.numeric import Inf
from tqdm import tqdm
import numpy as np
from numba import jit, prange
import argparse
from face_recognition import face_encodings

from sklearn.cluster import estimate_bandwidth, MeanShift
PREFIX_RETINA = 'models/retinaface-R50/R50'
RETINA_FOLDER = './insightface/detection/RetinaFace/'
sys.path.insert(1, RETINA_FOLDER)
from rcnn.dataset import retinaface
from retinaface import RetinaFace
import random

verbose = False

def get_annotations(image, detector):
    THRESH = 0.3
    DEFAULT_SCALES = [800, 1200]
    FLIP = True
    TEST_SCALES = [500, 800, 1100, 1400, 1700]
    ## Compute scale
    target_size = DEFAULT_SCALES[0]
    max_size = DEFAULT_SCALES[1]
    im_size_min = np.min(image.shape[0:2])
    im_size_max = np.max(image.shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    scales = [float(scale) / target_size * im_scale for scale in TEST_SCALES]
    ## Detection
    faces, _ = detector.detect(image, THRESH, scales=scales, do_flip=FLIP)

    return [{
        "x": int(float(box[0])),
        "y": int(float(box[1])),
        "w": int(float(box[2] - box[0])),
        "h": int(float(box[3] - box[1])),
        "p": float(box[4]),
        "box": [[int(float(box[i])) for i in range(4)]]
    } for box in faces]


def save_log(file, infos, prefix='', time=True):
    with open(file, "a") as log_file:
        for info in infos:
            time_prefix = datetime.now().strftime("%H:%M:%S ") if time else ''
            log_file.write(prefix + time_prefix + info + '\n')


def bbox_distance(a1, a2):
    x1 = a1['x'] + 0.5 * a1['w']
    x2 = a2['x'] + 0.5 * a2['w']
    y1 = a1['y'] + 0.5 * a1['h']
    y2 = a2['y'] + 0.5 * a2['h']
    return (x1 - x2)**2 + (y1 - y2)**2

def normalize(emb): 
    emb = np.array(emb)
    norm = np.sqrt(np.sum(emb*emb))
    return emb / norm if norm != 0 else 0

def clean_annotations(annotations, face_size_threshold = 0.5):
    error = False

    if len(annotations) > 0:
        ################ Remove unwanted faces ################
        # Remove low probability
        annotations = list(sorted(annotations, key=lambda a: -a['p']))
        if annotations[0]['p'] > 0.99:
            annotations = list(filter(lambda a: a['p'] > 0.99, annotations))
        else:
            annotations = [annotations[0]]
            error = True
        # Remove little BB
        max_area = max(annotations, key=lambda a: float(a['w'] * a['h']))
        max_area = max_area['w'] * max_area['h']
        if face_size_threshold > 1 :
            annotations = list(filter(lambda a: float(a['w']) > face_size_threshold, annotations))
        else :
            annotations = list(
                filter(lambda a: float(a['w'] * a['h']) > face_size_threshold * max_area,
                    annotations))
       

    else:
        error = True

    return (annotations, error)


def crop_image(im, x, y, w, h):
    f = 1.3
    (y_max, x_max, _) = im.shape
    # Use (u,v), center of the face
    # to simplify the calculations
    u, v = x + w // 2, y + h // 2
    w, h = int(f * w), int(f * h)
    w = min(u, x_max - u, w // 2)
    h = min(v, y_max - v, h // 2)
    x, y = u - w, v - h
    return im[v - h:v + h, u - w:u + w], x, y, 2 * w, 2 * h


def get_videos_infos(db: Cursor, dataset, method, compression, label):
    video_files = db.execute(
        (" SELECT videoid, dataset, folder, file, nb_frames,                                    "
         "        (SELECT group_concat(frame_nb,',') FROM frames f WHERE f.videoid = v.videoid) "
         " FROM videos v                                                                        "
         " WHERE ((SELECT count(1) FROM frames f WHERE f.videoid = v.videoid) < v.nb_frames - 1)"
         " AND (:dataset     IS NULL OR dataset     = :dataset)                                 "
         " AND (:compression IS NULL OR compression = :compression)                             "
         " AND (:method      IS NULL OR method      = :method)                                  "
         " AND (:label       IS NULL OR label       = :label)                                   "
         " ORDER BY RANDOM()                                                                    "
         ), {
             'dataset': dataset,
             'compression': compression,
             'method': method,
             'label': label
         }).fetchall()
    
    video_files_sorted = [video_file for video_file in video_files if video_file[1] == 'VideoConference'] + \
                         [video_file for video_file in video_files if video_file[1] != 'VideoConference']
    
    return video_files_sorted


def save_face_tmp(frame, image_path_tmp, annotation):
    face, x, y, w, h = crop_image(frame, annotation['x'], annotation['y'],
                                  annotation['w'], annotation['h'])

    makedirs(re.sub(r"/[^/]*$", "", image_path_tmp), exist_ok=True)
    cv2.imwrite(image_path_tmp, face)

    p = annotation['p']
    return face, x, y, w, h, p

def save_face(db, crop_result, image_folder, image_name, identity,
              frameid, image_path_tmp, output_folder):
    x, y, w, h, p = crop_result

    makedirs(join(output_folder, image_folder), exist_ok=True)
    move(image_path_tmp, join(output_folder, image_folder, image_name))

    db.execute(
        ("INSERT INTO faces(frameid, folder, file, identity, x, y, w, h, p)"
         "VALUES (:frameid, :folder, :file, :identity, :x, :y, :w, :h, :p) "),
        {
            'frameid': frameid,
            'folder': image_folder,
            'file': image_name,
            'identity': identity,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'p': p
        })

# @jit
def detect_crop_and_identify(dataset, method, compression, label, nb_frames_processed,
                             random_sample, db_folder, db_file, output_folder, face_size_threshold = None):

    detector = RetinaFace(PREFIX_RETINA, 0, 0, network='net3', vote=True)
    
    makedirs(db_folder, exist_ok=True)
    print("Saved in database : ", join(db_folder, db_file))
    conn = sqlite3.connect(db_folder + db_file)
    db = conn.cursor()

    video_files = get_videos_infos(db, dataset, method, compression, label)
    video_files = list(filter(lambda e: re.match(r"^laptop_a\d{4}_(in|out)door\.mp4$", e[3]) is None,
                              video_files)) # Filter out HCVM files (cf. face_detect_hcvm.py)

    skipped = []

    # for n in prange(len(video_files)):
        # videoid, dataset_folder, video_folder, video_file, nb_frames, frames = video_files[n]
    for (videoid, dataset_folder, video_folder, video_file, nb_frames,
         frames) in tqdm(video_files):

        # Create output folder
        video_name = re.sub(r"\.[^\.]*$", "", video_file)
        images_folder = join(dataset_folder, re.sub("/videos$", "",  video_folder), 'images', video_name)
        tmp_folder = output_folder + images_folder + "/tmp/"
        # if the folder already exist, we concidere the job has already be done
        # before and we skip to next iteration
        if exists(join(output_folder, images_folder)):
            skipped.append(video_file)
            continue

        makedirs(join(output_folder, images_folder), exist_ok=True)
        # print("Starting {} : {}/{}".format(video_file, n, len(video_files)))
        tqdm.write(f"Starting {video_file}")

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
        # elif dif_pre != "":
        #     already_processed_frames = set(
        #         map(lambda e: int(re.sub(r"[^/]*@\d\d:\d\d:\d\d_", "", basename(e)).split("_")[0]), 
        #              glob(join(images_folder,"**/*.*g"), recursive=True)))
        #     subset_frame = list(set(range(nb_frames)) - already_processed_frames)
        else:
            subset_frame = available_frames[:nb_frames_processed]

        # Load video
        video_path = db_folder + dataset_folder + '/' + video_folder + '/' + video_file
        video = cv2.VideoCapture(video_path)

        # Save frames
        annotations_all = []
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
                                        "embedding"  : face_encodings(face, face_boxes)[0]})

        # Cluster Identities
        with open(join(output_folder, images_folder, "annotations_all.pkl"), 'wb') as f: pickle.dump(annotations_all, f)
        # with open(join(output_folder, images_folder, "annotations_all.pkl"), 'rb') as f: annotations_all = pickle.load(f)
        if len(annotations_all) > 0:
            annotations_all = clusterFaces(annotations_all)
        else:
            warnings.warn("No faces were detected in : {}".format(video_file))
            continue

        image_names = []
        for i in range(len(annotations_all)):
            annotation     = annotations_all[i]["annotation"]
            frame_nb       = annotations_all[i]["frame_nb"]
            frameid        = annotations_all[i]["frameid"]
            identity       = annotations_all[i]["identity"]
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
        with open(join(output_folder, images_folder, "annotations_all.pkl"), 'wb') as f: pickle.dump(annotations_all, f)
        # print("Finished {} : {}/{}".format(video_file, n, len(video_files)))
        tqdm.write(f"Finished {video_file}")
        if len(listdir(tmp_folder)) < 1:
            rmdir(tmp_folder)
    conn.close()

    print("Skipped (Folder already exist) : \n")
    printCol(list(filter(lambda s: len(s) <= 30, skipped)), 3)
    printCol(list(filter(lambda s: len(s) >  30, skipped)), 2)


def printCol(lst: List[str], nb_col) :
    bigLst = []
    for i in range(0,len(lst),nb_col): 
        bigLst.append("\t".join(lst[i*nb_col:min((i+1)*nb_col,len(lst))]))
    print("\n".join(list(filter(lambda s: len(s) > 0, bigLst))))


def clusterFaces(annotations_all: List[dict]):
    # Extract the set of embeddings to so we can cluster on them
    embeddings = [a["embedding"] for a in annotations_all]

    # cluster the embeddings
    if verbose: tqdm.write("[INFO] clustering...")

    if verbose: tqdm.write("[INFO] Mean-Shift : Estimating Bandwidth")
    bandwidth = estimate_bandwidth(embeddings, quantile=0.3, n_samples=500)
    if bandwidth == 0 : bandwidth = 0.001

    if verbose: tqdm.write("[INFO] Mean-Shift : Initiating the model")
    clt = MeanShift(bandwidth=bandwidth, n_jobs=-1)

    if verbose: tqdm.write("[INFO] Mean-Shift : Fitting the model")
    clt.fit(embeddings)
        
    # determine the total number of unique faces found in the dataset
    # clt.labels_ contains the label ID for all faces in our dataset (i.e., which cluster each face belongs to).
    # To find the unique faces/unique label IDs, used NumPy’s unique function.
    # The result is a list of unique labelIDs
    labelIDs = np.unique(clt.labels_)

    # we count the numUniqueFaces . There could potentially be a value of -1 in labelIDs — this value corresponds
    # to the “outlier” class where a 128-d embedding was too far away from any other clusters to be added to it.
    # “outliers” could either be worth examining or simply discarding based on the application of face clustering.
    numUniqueFaces = len(np.where(labelIDs > -1)[0])
    if verbose: tqdm.write("[INFO] # unique faces: {}".format(numUniqueFaces))

    # loop over the unique face integers
    for labelID in labelIDs:
        # find all indexes into the `data` array that belong to the
        # current label ID, then randomly sample a maximum of 25 indexes
        # from the set
        # tqdm.write("[INFO] faces for face ID: {}".format(labelID))
        idxs = np.where(clt.labels_ == labelID)[0]

        for i in idxs:
            annotations_all[i]["identity"] = labelID
    return annotations_all


def parse_string(s, isFolder=False):
    if s == 'None':
        return None
    if isFolder and s[-1] != '/':
        return s + '/'
    return s


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
