# Adapted from https://github.com/kunalagarwal101/Face-Clustering/

# The revelant parts of this code were integrated in face_detect.py

# face_recognition library by @ageitgey
import face_recognition
# DBSCAN model for clustering similar embeddings
from sklearn.cluster import DBSCAN, KMeans, estimate_bandwidth, MeanShift

import argparse
import cv2
import numpy as np
from tqdm import tqdm
from os import listdir, makedirs
from os.path import isfile, join, basename
from shutil import copyfile
# import pickle
from six.moves import cPickle as pickle #for performance

def cropCenter(im: np.array, i: int):
    return im[im.shape[0]//i:-im.shape[0]//i, im.shape[1]//i:-im.shape[1]//i, :]

def getEmbeddings(datasetPath: str):
    # grab the paths to the input images in our dataset, then initialize
    # our data list (which we'll soon populate)
    print("[INFO] quantifying faces...")
    imageNames = [f for f in listdir(datasetPath) if isfile(join(datasetPath, f))]
    data = []

    # loop over the image paths
    for (imageName) in tqdm(imageNames, total=len(imageNames)):
        # load the input image and convert it from RGB (OpenCV ordering)
        # to dlib ordering (RGB)
        imagePath = join(datasetPath, imageName)

        # loading image to BGR
        image = cv2.imread(imagePath)

        # converting image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Use retina bounding boxes i.e. the image size
        boxes = [(0, image.shape[1], image.shape[0], 0)]

        # compute the facial embedding for the face
        embeddings = face_recognition.face_encodings(image, boxes)

        # build a dictionary of facial embeddings for the current image
        d = [{"imageName": imageName, "embedding": emb}
            for (box, emb) in zip(boxes, embeddings)]
        data.extend(d)
    return data


def move_image(image, imageName, labelID):
    path = "person_{:0>3}".format(labelID)

    # Ensure the specified path exists, create it if not.
    makedirs(path, exist_ok=True)

    # Saving the image
    cv2.imwrite(join(path, imageName), image)


def clusterFaces(data, datasetPath: str, outputPath="", method: str = 'DBSCAN', jobs=-1):
    # Extract the set of embeddings to so we can cluster on them
    imageNamesToConcider = [f for f in listdir(datasetPath) if isfile(join(datasetPath, f))]

    if len(data)<1 or not ("imageName" in data[0].keys() or "image_path" in data[0].keys()):
        print("[ERROR] Problem with the embeddings")
        return

    if "image_path" in data[0].keys():
        for d in data:
            d["imageName"] = basename(d["image_path"])

    data = [d for d in data if basename(d["imageName"]) in imageNamesToConcider]    
    data = np.array(data)
    embeddings = [d["embedding"] for d in data]

    # cluster the embeddings
    print("[INFO] clustering...")

    # creating DBSCAN object for clustering the embeddings with the metric "euclidean"
    if method == 'DBSCAN' :
        clt = DBSCAN(metric="euclidean", n_jobs=jobs)

    elif method.endswith('_mean'):
        k = int(method.split("_")[0])
        clt = KMeans(init="k-means++", n_clusters=k, n_init=4, random_state=0)
    
    elif method.endswith('mean-shift'):
        print("[INFO] Mean-Shift : Estimating Bandwidth")
        bandwidth = estimate_bandwidth(embeddings, quantile=0.3, n_samples=500)
        print("[INFO] Mean-Shift : Initiating the model")
        clt = MeanShift(bandwidth=bandwidth, n_jobs=jobs)
        print("[INFO] Mean-Shift : Fitting the model")
    
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
    print("[INFO] # unique faces: {}".format(numUniqueFaces))

    # loop over the unique face integers
    with tqdm(total=len(clt.labels_)) as pbar:
        for labelID in labelIDs:
            # find all indexes into the `data` array that belong to the
            # current label ID, then randomly sample a maximum of 25 indexes
            # from the set
            # print("[INFO] faces for face ID: {}".format(labelID))
            idxs = np.where(clt.labels_ == labelID)[0]

            for i in idxs:
                pbar.update(1)
                
                srcPath = join(datasetPath , data[i]["imageName"])
                idPath  = join(outputPath, "{:0>3}".format(labelID))
                dstPath = join(idPath , data[i]["imageName"])

                # Ensure the specified path exists, create it if not.
                makedirs(idPath, exist_ok=True)

                copyfile(srcPath, dstPath)


if __name__ == "__main__":
    
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--dataset", required=True,
        help="path to input directory of faces + images")
    ap.add_argument("-o", "--output", type=str, default=None,
        help="path to the output directory (where to put the clustered faces)")
    ap.add_argument("-m", "--method", type=str, default="mean-shift",
        help="Clustering method ('DBSCAN', 'mean-shift', '{k}_mean' (replace {k} by the number of class, e.g. '2_mean'))")
    ap.add_argument("-s", "--save", action='store_true',
        help="Save the embeddings ?")
    ap.add_argument("-f", "--fromFile", action='store_true',
        help="Use an embeddings file")
    ap.add_argument("-e", "--embeddings", type=str, default="embeddings.pickle",
        help="Name of th embeddings")
    ap.add_argument("-j", "--jobs", type=int, default=-1,
        help="# of parallel jobs to run (-1 will use all CPU)")
    args = vars(ap.parse_args())

    if not args["fromFile"]:
        data = getEmbeddings(args["dataset"])
    else :
        # data = pickle.loads(open(args["embeddings"], "rb").read())
        with open(args["embeddings"], 'rb') as f: data = pickle.load(f)

    if args["save"] and not args["fromFile"]:
        makedirs(args["output"], exist_ok=True)
        f = open(join(args["output"], args["embeddings"]), "wb")
        f.write(pickle.dumps(data))
        f.close()

    if args["output"] is None:
        output = args["method"]
    else :
        output = args["output"]

    clusterFaces(data, args["dataset"], output, args["method"], args["jobs"])

# py cluster_faces.py -i . -s 
# py cluster_faces.py -i . -m 2_mean -f True 
