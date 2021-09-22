from sqlite3.dbapi2 import Cursor
import sqlite3
import cv2
from os import listdir, makedirs
from os.path import isfile, join
import re
import argparse
from tqdm import tqdm
from typing import List
import time, random

def getCropInfo(db: Cursor, frameId: int):
    crop_info = db.execute(
        " SELECT faceid, frameid, folder, file, identity, x, y, w, h, p FROM faces f WHERE (f.frameid = :frameId);", 
        { 'frameId': frameId }
        ).fetchall()

    return [{"faceid"   : crop_info[i][0],
             "frameid"  : crop_info[i][1],
             "folder"   : crop_info[i][2],
             "file"     : crop_info[i][3],
             "identity" : crop_info[i][4],
             "x"        : crop_info[i][5],
             "y"        : crop_info[i][6],
             "w"        : crop_info[i][7],
             "h"        : crop_info[i][8],
             "p"        : crop_info[i][9],
            } for i in range(len(crop_info))]

def addFakeVideo(db: Cursor, original_videoid: int, fake_vid_folder: str, fake_vid_file: str):
    """
        Copy the attributes of the real video, modifying them when it's revealant.
        Return the 'videoid' of the fake video
    """

    print(original_videoid, fake_vid_folder, fake_vid_file)
    
    db.execute("CREATE TEMPORARY TABLE video_tmp AS SELECT * FROM videos WHERE videoid = :origId;", { 'origId': original_videoid })
    db.execute("UPDATE video_tmp SET videoid = NULL;")
    db.execute("UPDATE video_tmp SET label   = 'FAKE';")
    db.execute("UPDATE video_tmp SET folder  = :folder;", {'folder': fake_vid_folder})
    db.execute("UPDATE video_tmp SET file    = :file;", {'file': fake_vid_file})
    db.execute("INSERT INTO videos SELECT * FROM video_tmp;")
    db.execute("DROP TABLE video_tmp;")

    return db.execute("SELECT last_insert_rowid()").fetchone()[0]

def addFrame(db: Cursor, videoid: int, frame_nb: int):
    """
        Add a frame to the database 'db'.
        Return the 'frameid'
    """
    db.execute("INSERT INTO frames(videoid,frame_nb) VALUES (?,?)", (videoid, frame_nb))
    return db.execute("SELECT last_insert_rowid()").fetchone()[0]

# def addFakeFrames(db, original_videoid: int, fake_videoid: int):
#     # We copy the attributes of the real video frames, modifying them when it's revealant
#     db.executescript((
#         "CREATE TEMPORARY TABLE frames_tmp AS SELECT * FROM frames WHERE videoid={origId:d};"
#         "UPDATE frames_tmp SET frameid = NULL;        "
#         "UPDATE frames_tmp SET videoid = {fakeId:d};  "
#         "INSERT INTO frames SELECT * FROM frames_tmp; "
#         "DROP TABLE frames_tmp;                       "
#         ).format(origId= original_videoid, fakeId=fake_videoid))
#     return db.execute("SELECT last_insert_rowid()").fetchone()

def addFakeFace(db: Cursor, cropInfo: dict, frameid: int, method: str, 
                donor: dict, receiver: dict,
                image_folder: str, image_name: str):

    donor_videoid = donor["videoid"]
    receiver_videoid = receiver["videoid"]
    donor_identity = donor["identity"]

    exist = db.execute("SELECT faceid FROM faces WHERE(folder = :folder AND file = :file);"
            , {
                'folder': image_folder,
                'file': image_name,
            }).fetchone()

    if exist is None:

        db.execute(
            ("INSERT INTO faces(frameid, folder, file, identity, x, y, w, h, p, isfake, method, donor_videoid, donor_identity, receiver_videoid, receiver_identity)"
            "VALUES (:frameid, :folder, :file, :identity, :x, :y, :w, :h, :p, :isfake, :method, :donor_videoid, :donor_identity, :receiver_videoid, :receiver_identity)"),
            {
                'frameid' : frameid,
                'folder'  : image_folder,
                'file'    : image_name,
                'identity': cropInfo["identity"],
                'x': cropInfo["x"],
                'y': cropInfo["y"],
                'w': cropInfo["w"],
                'h': cropInfo["h"],
                'p': cropInfo["p"],
                'isfake': 1,
                'method': method,
                'donor_videoid'    : donor_videoid,
                'donor_identity'   : donor_identity,
                'receiver_videoid' : receiver_videoid,
                'receiver_identity': cropInfo["identity"],
            })
def get_video_name(path: str):
    res = re.search(r"(/|^)[^/]*@\d\d:\d\d:\d\d", path)
    if res is None:
        res = re.search(r"(/|^)laptop_a\d{4}_(in|out)door", path)
    res = res.group()
    if res[0] == "/" :
        res = res[1:]
    return res

def parseNameAndIdentity(db: Cursor, extract_and_identity: str, dataset: str, original_video_folder: str):
    extract_name = get_video_name(extract_and_identity)
    identity     = re.search(extract_name + r"_\d{3}$", extract_and_identity).group()[-3:]

    videoid_nb_frames = db.execute(
        "SELECT videoid, nb_frames, file FROM videos v WHERE (v.dataset = :dataset AND v.folder = :folder) AND (v.file LIKE :file)",
    {
        "dataset": dataset,
        "folder": original_video_folder,
        "file": extract_name + ".%",
    }).fetchone()
    return {"video_name": extract_name,
            "extension" : videoid_nb_frames[2][len(extract_name):], # The file extension (most likely mp4)
            "identity"  : identity, 
            "videoid"   : videoid_nb_frames[0],
            "nb_frames" : videoid_nb_frames[1],
            }

def getFrameIdsDict(db: Cursor, videoid, faces_files_name: List[str]):
    """
        Return a dictionary with 'frame_nb' as keys and 'frameid' as values for all revealant frames
        (e.g. {2: 3} if frame_nb = 2 and frameid = 3)
    """
    # The following is not proprely sanitized but for the use case it should be OK
    # (sqlite3 appears not to allow the insertion of a list through a placeholder)
    requestFakeFrameNbs = " SELECT group_concat(frame_nb,','), group_concat(frameid,',') FROM frames fr " + \
                          " WHERE fr.videoid = {} AND fr.frameid IN " + \
                          " (SELECT frameid FROM faces f WHERE (f.file IN ({})))"
    frameNbsAndIds = db.execute(requestFakeFrameNbs.format(videoid, str(faces_files_name)[1:-1])).fetchone()

    frameNbs = [int(ff) for ff in frameNbsAndIds[0].split(",")]
    frameIdsList = [int(ff) for ff in frameNbsAndIds[1].split(",")]
    frameIdsDict = {frameNbs[i]: frameIdsList[i] for i in range(len(frameNbs))}

    return frameIdsDict


def insertAllFakes(fakes_path: str, method: str, dataset: str):
    fakes_folders = sorted(listdir(fakes_path))

    conn = sqlite3.connect(DB_FOLDER + DB_FILE, timeout=300)
    db = conn.cursor()
    # cursor=db.execute("SELECT * FROM faces")
    # names = list(map(lambda x: x[0], cursor.description))

    # TODO Here we make the hypothesis that the original video was real
    # Some bug may occur if it's wrong
    original_video_folder = "{}-real/videos".format(dataset)

    for fake_folder in fakes_folders:
        extracts_and_identities = fake_folder.split('⟶')
        receiver = parseNameAndIdentity(db, extracts_and_identities[0], dataset, original_video_folder)
        donor    = parseNameAndIdentity(db, extracts_and_identities[1], dataset, original_video_folder)
        insertDeepFake(db, method, dataset, join(fakes_path, fake_folder), original_video_folder, donor, receiver)
        conn.commit()
    conn.close()

def insertDeepFake(db: Cursor, method: str, dataset: str, fakes_path: str, original_video_folder: str, 
                   donor: dict, receiver: dict):
    """
        Insert the files in `fakes_path` back into the video they originally
        comes from.
    """

    # Check deepfakes files
    files = [f for f in listdir(fakes_path) if isfile(join(fakes_path, f))]
    associatedFiles = [re.sub(r"\.[^\.]*$", ".jpg", f) for f in files] # All our files from retina are jpg

    # Create output folder
    out_path = "{0}/{0}-synthesis/videos/{1}/{2}/".format(dataset, method, receiver["video_name"])
    makedirs(out_path, exist_ok=True)

    # Open the original video and get the data needed for the creation of videoFake
    videoOriginal = cv2.VideoCapture("{}/{}/{}".format(dataset, original_video_folder, receiver["video_name"] + receiver["extension"]))
    width     = int(videoOriginal.get(cv2.CAP_PROP_FRAME_WIDTH))
    height    = int(videoOriginal.get(cv2.CAP_PROP_FRAME_HEIGHT))
    framerate = int(videoOriginal.get(cv2.CAP_PROP_FPS))

    # Create videoFake with an unique, meaningful name
    vidR = receiver["video_name"] + "_" + receiver["identity"]
    vidD = donor["video_name"]    + "_" + donor["identity"]
    vidFakeName = f"{vidR}⟶{vidD}-{method}"
    vidFakeNb = len([f for f in listdir(out_path) if isfile(join(out_path, f)) and f.startswith(vidFakeName)])
    vidFakeName += (f"-{vidFakeNb:0>3}" if vidFakeNb>0 else "") + receiver["extension"]
    # vidFakeName = "{}-{}-{:0>4}{}".format(receiver["video_name"], method, vidFakeNb, receiver["extension"])

    fake_videoid = addFakeVideo(db, receiver["videoid"], out_path, vidFakeName)
    
    videoFake = cv2.VideoWriter(out_path + vidFakeName, cv2.VideoWriter_fourcc('M','P','4','V'), framerate, (width, height))
    # videoFake = cv2.VideoWriter(out_path + vidFakeName, cv2.VideoWriter_fourcc( *"WMV2" ), framerate, (width, height))

    #Retrieve
    receiverFrameIds = getFrameIdsDict(db, receiver["videoid"], associatedFiles)


    i = -1
    with tqdm(total=receiver["nb_frames"]) as pbar:  
        while True:
            ret, frame = videoOriginal.read()
            i += 1
            pbar.update(1)

            if ret == True:
                if i in receiverFrameIds.keys():
                    crops = getCropInfo(db, receiverFrameIds[i])
                    fakeFrameid = addFrame(db, fake_videoid, i)

                    for crop in crops:
                        if crop["file"] in associatedFiles:
                            file = files[associatedFiles.index(crop["file"])]
                            im = cv2.imread(join(fakes_path, file))
                            
                            frame[crop["y"]:crop["y"] + crop["h"], 
                                  crop["x"]:crop["x"] + crop["w"],
                                  :] = im

                            addFakeFace(db, crop, fakeFrameid, method, donor, receiver, fakes_path, file)
                    
                # Write the frame into the file
                videoFake.write(frame)

            # Break the loop
            else:
                break

	# When everything done, release the video capture and video write objects
    videoOriginal.release()
    videoFake.release()


def parse_string(s, isFolder=False):
    if s == 'None':
        return None
    if isFolder and s[-1] != '/':
        return s + '/'
    return s


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--db",
                        default='./',
                        help="Path to the database")
    parser.add_argument("--dbFile",
                        default="Deepfake.db",
                        help="Path to the database (.db file)")
    parser.add_argument("-i", "--input",
                        default="VideoConference/VideoConference-synthesis/images/deepfakes_faceswap",
                        type=str,
                        help="Input folder (should contain folders for the different fakes")
    parser.add_argument("--dataset",
                        default='VideoConference',
                        type=str,
                        help="The concerned dataset")
    parser.add_argument("-m", "--method",
                        default='deepfakes_faceswap',
                        type=str,
                        help="Which method was used to create the fake")
    args = parser.parse_args()
    DB_FOLDER = parse_string(args.db, isFolder=True)
    DB_FILE = parse_string(args.dbFile)
    IN_FOLDER = DB_FOLDER + parse_string(args.input)
    METHOD = parse_string(args.method)
    DATASET = parse_string(args.dataset)

    insertAllFakes(IN_FOLDER, METHOD, DATASET)
    # insertDeepFake(db, IN_FOLDER, METHOD)
    

# cd ~/PFE/video-conference-dataset/datasets
# py face_reinsertion.py -i ../data/out/nicolas_wassim/ -m deepfakes_faceswap
# py face_reinsertion.py -i VideoConference/VideoConference-synthesis/images/deepfakes_faceswap -m deepfakes_faceswap
# py face_reinsertion.py -i VideoConference/VideoConference-synthesis/images/faceshifter -m faceshifter