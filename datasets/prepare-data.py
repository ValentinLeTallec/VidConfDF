import sqlite3
import os
from shutil import copyfile
import argparse

def parse_string(s, is_folder=False):
    if s == 'None':
        return None
    if is_folder and s[-1] != '/':
        return s + '/'
    return s

def create_split(db, train_p, test_p, valid_p):
    pass

def copy_images(faces, split, image_folder, output_folder):
    from_split = list(filter(lambda x: x[1] == split, faces))
    real_faces = list(filter(lambda x: x[2] == 'REAL', from_split))
    fake_faces = list(filter(lambda x: x[2] == 'FAKE', from_split))

    if not os.path.exists(output_folder + split.lower() + '/fake'):
        os.makedirs(output_folder + split.lower() + '/fake')
    for i, p in enumerate(fake_faces):
        copyfile(image_folder + p[0],
                 output_folder + split.lower() + '/fake/' + str(i) + '.jpeg')
    if not os.path.exists(output_folder + split.lower() + '/real'):
        os.makedirs(output_folder + split.lower() + '/real')
    for i, p in enumerate(real_faces):
        copyfile(image_folder + p[0],
                 output_folder + split.lower() + '/real/' + str(i) + '.jpeg')
    print("Copied " + str(len(fake_faces) + len(real_faces))  +
          "images into " + split.lower() + " folder")

def save_images(db, args, image_folder, output_folder):
    faces = db.execute((
        "WITH  single_face AS (SELECT videoid                                                     "
        "                      FROM videos v                                                      "
        "                      WHERE :single = 0                                                  "
        "                      OR    NOT EXISTS (SELECT 1                                         "
        "                                        FROM frames fr                                   "
        "                                        WHERE fr.videoid = v.videoid                     "
        "                                        AND (SELECT COUNT(1)                             "
        "                                             FROM faces fa                               "
        "                                             WHERE fa.frameid = fr.frameid) > 1)),       "
        "      ffpp_frames AS (SELECT v.videoid, frameid, frame_nb, split, label,                 "
        "                             ROW_NUMBER() OVER (PARTITION                                "
        "                                                BY v.videoid                             "
        "                                                ORDER BY RANDOM()) as rn                 "
        "                      FROM frames f                                                      "
        "                      JOIN videos v    USING (videoid)                                   "
        "                      JOIN single_face USING (videoid)                                   "
        "                      JOIN video_split vs ON vs.videoid = v.videoid AND origin = :split  "
         "                     WHERE (:dataset     IS NULL OR dataset     = :dataset)             "
         "                     AND   (:compression IS NULL OR compression = :compression)         "
         "                     AND   (:method      IS NULL OR (method = :method OR method = '' )) "
        "                      ORDER BY v.videoid)                                                "
        " SELECT  (fa.folder || '/' || fa.file), split, label                                     "
        " FROM faces fa                                                                           "
        " JOIN ffpp_frames f ON f.frameid = fa.frameid                                            "
        " WHERE f.rn <= :nb_frames                                                                "
        " ORDER BY f.videoid, f.rn                                                                "
    ), args).fetchall()

    copy_images(faces, 'TRAIN', image_folder, output_folder)
    copy_images(faces, 'TEST', image_folder, output_folder)
    copy_images(faces, 'VALID', image_folder, output_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o",
                        "--output",
                        help="<REQUIRED> Path to the output folder")
    parser.add_argument("-s",
                        "--split",
                        default='(0.75,0.15,0.15)',
                        help="Which split to use")
    parser.add_argument("-n",
                        "--num",
                        default=10,
                        type=int,
                        help="Number of frames per videos")
    parser.add_argument("--db", default='.', help="Path to the database")
    parser.add_argument("--dbFile",
                        default="Deepfake.db",
                        help="Path to the database (.db file)")
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
    parser.add_argument("-c",
                        "--compression",
                        default="None",
                        type=str,
                        help="Which compression to be considered")
    parser.add_argument("--single",
                        default=True,
                        type=bool,
                        help="Single face per frames")
    args = parser.parse_args()

    DB_FOLDER = parse_string(args.db, is_folder=True)
    OUTPUT_FOLDER = parse_string(args.output, is_folder=True)
    DB_FILE = parse_string(args.dbFile)
    query_args = {
        'dataset':parse_string(args.dataset),
        'method':parse_string(args.method),
        'compression':parse_string(args.compression),
        'split': args.split,
        'nb_frames': args.num,
        'single': args.single
    }
    conn = sqlite3.connect(DB_FOLDER + DB_FILE)
    db = conn.cursor()
    save_images(db,query_args, DB_FOLDER, OUTPUT_FOLDER)
    conn.close()
