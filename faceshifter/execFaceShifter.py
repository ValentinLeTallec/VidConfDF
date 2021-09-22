from os.path import join, exists, expanduser, basename, isfile, isdir
from os import makedirs, listdir, rmdir
from six.moves import cPickle as pickle #for performance
from faceshifter_convert import faceShifter 
import argparse
import random
from tqdm import tqdm
from typing import List
import re
import csv

simulate = False

def get_subfolders(rootFolder: str):
    return [join(rootFolder, f) for f in listdir(rootFolder) if isdir(join(rootFolder, f))]


def get_images(rootFolder: str):
    isImage = lambda path: isfile(path) and not (re.search(r"\.??g$", path) is None)
    return [join(rootFolder, f) for f in listdir(rootFolder) if isImage(join(rootFolder, f))]

def get_video_name(path: str):
    res = re.search(r"(/|^)[^/]*@\d\d:\d\d:\d\d", path)
    if res is None:
        res = re.search(r"(/|^)laptop_a\d{4}_(in|out)door", path)
    res = res.group()
    if res[0] == "/" :
        res = res[1:]
    return res


def genPairs(images_folder: str):
    possibilities = []
    folders = get_subfolders(images_folder)

    for folder in folders:
        subfolders = get_subfolders(folder)

        # Only use rightly identified faces
        if any([basename(s) == "manual" for s in subfolders]):
            folder = join(folder, "manual")
            subfolders = get_subfolders(folder)
        elif "bad" in listdir(folder):
            continue

        subfolders = list(filter(lambda e: not e.endswith("_faces"), subfolders))
        for subfolder in subfolders:
            if basename(subfolder) != "tmp":
                possibilities.append(subfolder)

    random.shuffle(possibilities)
    if len(possibilities) % 2 == 1:
        rndIdx = random.randint(0, len(possibilities)-2)
        possibilities.append(possibilities[rndIdx])

    stop = False
    while not stop:
        stop = True
        for i in range(len(possibilities)-1):
            if get_video_name(possibilities[i]) == get_video_name(possibilities[i+1]):
                if i < len(possibilities)-2:
                    possibilities_end = possibilities[i+1:]
                    random.shuffle(possibilities_end)
                    possibilities = possibilities[:i+1] + possibilities_end
                else:
                    possibilities = [possibilities[-1]] + possibilities[:-1]

                stop = False


    return possibilities

def faceShifterOnFolder(receiver_dir: str, donor_dir: str, out_path: str, k_max = 42):

    donor_image = random.choice(get_images(donor_dir))
    receiver_files = [join(receiver_dir, f) for f in listdir(receiver_dir) if isfile(join(receiver_dir, f)) 
                      and not f.endswith(".fsa")
                      and not f.endswith(".pkl")
                      and not f in ["good", "bad"]]

    faceShifterSimple = lambda receiver, donor: faceShifter(receiver, donor, 
                                                            # join(out_path, '{receiver}⟶{donor}'), 
                                                            join(out_path, f'{receiver_dir}_{donor_dir}'), 
                                                            '{receiver_file}', throwError=False)

    for receiver in receiver_files:
        res = faceShifterSimple(receiver, donor_image)
        
        # If our first choice of donor image didn't work then we try another
        k = 0
        while res == "errDonor" and k < k_max:
            k += 1
            donor_image = random.choice(get_images(donor_dir))
            res = faceShifterSimple(receiver, donor_image)

        # If too much iteration pass we skip this combination
        if k == k_max:
            print("{} unsuccessful attempt at using {} files as donor, skipping".format(k_max, donor_dir))
            break


def applyFaceShifter(possibilities: List[str], out_path: str, go_both_ways = True):

    for i in tqdm(range(0, len(possibilities), 2)):
        id1 = possibilities[i]
        id2 = possibilities[i+1]

        if simulate:
            id1_image = random.choice(get_images(id1))
            id2_image = random.choice(get_images(id2))

            print("{}\n{}\n".format(id1_image, id2))
            print("{}\n{}\n".format(id2_image, id1))
        else:
            faceShifterOnFolder(id1, id2, out_path)
            if go_both_ways:
                faceShifterOnFolder(id2, id1, out_path)

def genPairPickle(images_folder: str, file_name = "pairs.pkl"):
    possibilities = genPairs(images_folder)
    possibilities = [basename(s[:s.find(get_video_name(s))-1]) + s[s.find(get_video_name(s))-1:] for s in possibilities]
    pairs = [{"donor": possibilities[i], 
              "receiver": possibilities[i+1]} for i in tqdm(range(0, len(possibilities), 2))]
    for p in pairs:
        donor = "_".join(p["donor"].split("/")[1:])
        receiver = "_".join(p["receiver"].split("/")[1:])
        p["out_folder"] = f"hififace/{receiver}⟶{donor}"
        
    with open(join(images_folder, '..', file_name), 'wb') as f: pickle.dump(pairs, f)
    # with open(join(images_folder, file_name), 'rb') as f: pairs = pickle.load(f)
    print('')

if __name__ == "__main__":

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--folder",
                    default=expanduser("video-conference-dataset/datasets/VideoConference/VideoConference-real/images"),
                    help="Folder containing the photos' folders")
    ap.add_argument("-p", "--pairs",
                    default=None,
                    help=("A csv file containing one pair per line, with each line being `/path/to/receiver_folder,/path/to/donor_folder`"
                          "Note that if no file is defined the pairs will be generated randomly and go both ways"))
    ap.add_argument("-o", "--output", type=str,
                    default=expanduser("video-conference-dataset/datasets/VideoConference/VideoConference-synthesis/images/faceshifter/"),
                    help="path to the output directory")
    ap.add_argument("--simulate", "-s", action='store_true', 
                    help="Add this option to simulate (i.e. It won't spawn the commands)")
    args = vars(ap.parse_args())

    images_folder = args["folder"]
    out_path = args["output"]
    simulate = args["simulate"]
    pairsFile = args["pairs"]

    if pairsFile is None:
        possibilities = genPairs(images_folder)
        applyFaceShifter(possibilities, out_path)
    else:
        possibilities = []
        with open(pairsFile, 'r') as file:
            for line in csv.reader(file):
                possibilities += line
        applyFaceShifter(possibilities, out_path, False)
    print(possibilities)

# py /video-conference-dataset/faceshifter/execFaceShifter.py -f datasets/VideoConference/VideoConference-real/images -o datasets/VideoConference/VideoConference-synthesis/images/faceshifter/
