from os import listdir, makedirs
from os.path import isfile, join
import sqlite3
import re
import cv2
from tqdm import tqdm

datasetName = 'VideoConference_extended'

def get_meta(video):
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    nb_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    return {
        'w': width,
        'h': height,
        'nb_frames': nb_frames,
    }


folders = [
    (datasetName + '-real/videos', True, ''),
  # (datasetName + '-synthesis/videos', False, 'deepfakes_faceswap'),
]
# TODO if needed : complete the implementation for fakes videos

if __name__ == '__main__':

    conn = sqlite3.connect('../Deepfake.db')
    db = conn.cursor()

    for (folder, is_real, method) in tqdm(folders):
        makedirs(folder, exist_ok=True)
        files = [f for f in listdir(folder) if isfile(join(folder, f))]
        label = 'REAL' if is_real else 'FAKE'
        for file in tqdm(files):
            cap = cv2.VideoCapture(join(folder, file))
            meta = get_meta(cap)

            if meta["w"] == 0 or meta["h"] == 0 or meta["nb_frames"] == 0 :
                print(">>> Skipping {} because w = {},  h = {} and nb_frames = {}".format(
                    file, meta["w"], meta["h"], meta["nb_frames"]))
                continue

            if label == 'REAL':
                source_file, target_file = '', ''
            elif label == 'FAKE':
                parts = file.split('-')
                method = parts[1]
                source_file = parts[0] + "." + parts[2].split('.')[-1]
                target_file = source_file

            db.execute(
                ("INSERT INTO videos( label, folder, file, dataset, width, height, nb_frames )"
                 " VALUES ( :label, :folder, :file, :datasetName, :w, :h, :nb_frames)         "
                 " ON CONFLICT DO NOTHING                                                     "
                 ),
                {
                    'label': label,
                    'folder': folder,
                    'file': file,
                    'datasetName': datasetName,
                    # Metadata
                    'w': meta['w'],
                    'h': meta['h'],
                    'nb_frames': meta['nb_frames'],
                })

    conn.commit()
    conn.close()
