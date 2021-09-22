"""This version concidere that the manual classification is in a `manual` folder"""

from os import listdir, makedirs, rmdir
from os.path import isfile, join, basename, isdir, dirname, exists
from shutil import rmtree, move
import re
import argparse
from sqlite3.dbapi2 import Cursor
import sqlite3
import glob

verbose = False
very_verbose = False
rename_synthesis_files = True  # TODO: put it to false

def get_subfolders(rootFolder: str):
    return [join(rootFolder, f) for f in listdir(rootFolder) if isdir(join(rootFolder, f))]


def basename2(folder: str):
    if basename(folder) == "":
        folder = dirname(folder)
    return basename(folder)


def is_identity_folder(folder: str):
    folder = basename2(folder)
    return len(folder) == 3 and folder.isdigit()


def is_face_folder(folder: str):
    folder = basename2(folder)
    return len(folder) == 9 and folder[:3].isdigit() and folder.endswith("_faces")


def get_images(rootFolder: str):
    def isImage(path): return isfile(path) and not (
        re.search(r"(?i)\.(png|jpg|jpeg)$", path) is None)
    return [join(rootFolder, f) for f in listdir(rootFolder) if isImage(join(rootFolder, f))]


def get_video_name(path: str):
    regexpYoutube = r"(/|^)[^/]*@\d\d:\d\d:\d\d"
    regexpHCVM = r"(/|^)laptop_a\d{4}_(in|out)door"
    regexp = f"({regexpYoutube}|{regexpHCVM})" + r"(_extended)?"
    res = re.search(regexp, path).group()
    if res[0] == "/":
        res = res[1:]
    return res


def manual_clustering_folder(video_folder: str, expect_manual_folder: bool):
    """
        Arguments:
            - video_folder : path to the folder containing all face images of a video
            - expect_manual_folder : was the manual clustering done in a separated `manual` folder ?
        Return:
            - manual_subfolders : the subfolders ("/path/to/000", "/path/to/001", ...) to the manual clustering
    """
    video_subfolders = get_subfolders(video_folder)

    is_bad_clustering = ("bad" in listdir(video_folder))
    manual_exist = any([basename(s) == "manual" for s in video_subfolders])

    should_be_updated = ((not is_bad_clustering) and
                         ((expect_manual_folder and manual_exist) or
                          (not expect_manual_folder and not manual_exist)))

    if should_be_updated:
        if expect_manual_folder:
            manual_subfolders = get_subfolders(join(video_folder, "manual"))
        else:
            manual_subfolders = video_subfolders

    else:
        manual_subfolders = []

    manual_subfolders = list(filter(is_identity_folder, manual_subfolders))

    return manual_subfolders


def update_with_manual_clustering(db: Cursor, root: str, expect_manual_folder: bool):
    """
        Arguments:
            - root : folder containing all the {video_folder}s
                - video_folder : path to the folder containing all face images of a video in subfolders
            - expect_manual_folder : was the manual clustering done in a separated `manual` folder ?
    """

    # tmp prefix to ensure there is no conflict between new and original paths
    prefix = "[manual]"

    for video_folder in get_subfolders(root):
        video_subfolders = get_subfolders(video_folder)

        manual_subfolders = manual_clustering_folder(
            video_folder, expect_manual_folder)
        should_be_updated = len(manual_subfolders) > 0

        if should_be_updated:

            # Update the database
            for identity_folder in manual_subfolders:
                update_folder_with_manual_clustering(
                    db, identity_folder, prefix)

            # Update the filesystem
            if expect_manual_folder:
                # Remove the automatic clustering files
                for sub in [v for v in video_subfolders if is_identity_folder(v) or is_face_folder(v)]:
                    if verbose:
                        print(f"REMOVE FOLDER : {sub}")
                    rmtree(sub)
                if verbose:
                    print('-'*80)

                # Move the manually clustered folders to video_folder except for those which are not faces
                for identity_folder in [v for v in manual_subfolders if is_identity_folder(v) or is_face_folder(v)]:
                    if identity_folder.endswith("/999") or identity_folder.endswith("/999_faces"):
                        # Those images are not faces
                        if verbose:
                            print(f"REMOVE FOLDER : {identity_folder}")
                        rmtree(identity_folder)
                    else:
                        if verbose:
                            print(f"MOVE TO PARENT FOLDER : {identity_folder}")
                        move(identity_folder, video_folder)
                if verbose:
                    print('#'*80)

                # Remove the empty `manual` folder
                rmdir(join(video_folder, "manual"))

            # Remove prefix from filenames
            for identity_folder in list(filter(is_identity_folder, get_subfolders(video_folder))):
                rm_filename_prefix_in_folder(prefix, identity_folder)
                if exists(identity_folder + "_faces"):
                    rm_filename_prefix_in_folder(
                        prefix, identity_folder + "_faces")
                if rename_synthesis_files:
                    a = glob.glob(
                        f"VideoConference/VideoConference-synthesis/images/**/?{prefix[1:]}*")
                    if len(a) > 0:
                        print(a)
                    # print(e)
                    for e in glob.glob(f"VideoConference/VideoConference-synthesis/images/**/?{prefix[1:]}*", recursive=True):
                        move(e, join(dirname(e), basename2(e)[len(prefix):]))

    # Remove prefix from database
    db.execute("UPDATE faces SET folder = SUBSTR(folder,:len) WHERE folder LIKE :prefix;", {
        'len': len(prefix) + 1,
        'prefix': prefix + "%",
    })


def rm_filename_prefix_in_folder(prefix: str, folder: str):
    files_with_prefix = list(
        filter(lambda f: f.startswith(prefix), listdir(folder)))
    for file in files_with_prefix:
        move(join(folder, file), join(folder, file[len(prefix):]))

# file = pt-wO73Y8to@00:09:15_extended_01238_002.jpg
# VideoConference_extended/VideoConference_extended-real/images/pt-wO73Y8to@00:09:15_extended/002
# image = join(identity_folder, "pt-wO73Y8to@00:09:15_extended_01238_002.jpg")


def update_folder_with_manual_clustering(db: Cursor, identity_folder: str, prefix: str):
    # Get the manually assigned identity
    identity_manual = basename2(identity_folder)

    # If it's not what we expect we quit
    if not is_identity_folder(identity_manual):
        return

    if verbose:
        print(f"\n>>> Processing : {identity_folder}")
    video_name = get_video_name(identity_folder)

    used_file_names = []
    for image in get_images(identity_folder):

        # Extract all info from the original file name
        original_file = basename(image)
        extension = re.search(r"\.(png|jpg|jpeg)$", original_file).group()

        im_info = original_file[len(video_name)+1: - len(extension)].split("_")
        frame_nb = im_info[0]
        # identity_auto = im_info[1]

        # Determine the original and new folder
        base_folder = identity_folder[:identity_folder.index(
            video_name)+len(video_name)]
        # original_folder = join(base_folder, identity_auto)
        new_folder = join(base_folder, identity_manual)

        # Determine the new file name
        new_file = f"{video_name}_{frame_nb}_{identity_manual}{extension}"

        # Manage de possibility that multiple face of the same frame
        # are concidered as having the same identity
        # (while the name already exist : create new name)
        j = 0
        while new_file in used_file_names:
            new_file = f"{video_name}_{frame_nb}_{identity_manual}_{j:0>3}{extension}"
            j += 1

        # Search for the face in the database
        # We can accurately guess the exact `original_folder` because there are some non-standard `folder` such as :
        # - "path/to/video_folder/_000"
        # - "VideoConference/VideoConference-real/hcvm/000"
        # As such, we stay more generic.
        # However there should still only be one match as filenames are unique within real images
        faceid = db.execute(('SELECT faceid FROM faces '
                             'WHERE file = :original_file AND '
                             '((folder LIKE "VideoConference/VideoConference-real/hcvm/000") OR '
                             ' (folder LIKE "VideoConference/VideoConference-real/images/%") OR '
                             ' (folder LIKE "VideoConference_extended/VideoConference_extended-real/images/%"))'),
                            {'original_file': original_file}).fetchall()

        if len(faceid) != 1:
            print(
                f"ERROR : Found {len(faceid)} matches (expected 1) for the following file")
            print(
                f"        (the prefix '[ERROR]' was added to the file's name to prevent conflicts) :")
            print(f"> {original_file}")
            move(image, join(dirname(image), "[ERROR]" + new_file))
            continue

        faceid = faceid[0][0]

        if identity_manual == "999":  # it means those are not faces images
            # Remove them from the database
            if verbose:
                print(f"DELETE : {original_file}")
            db.execute("DELETE FROM faces WHERE faceid = :faceid;",
                       {'faceid': faceid})

        else:
            # Update the database
            if very_verbose and original_file != new_file:
                print(f"UPDATE : {original_file} TO {new_file}")
            used_file_names.append(new_file)

            db.execute(("UPDATE faces "
                        "SET folder = :new_folder, file = :new_file, identity = :new_identity "
                        'WHERE faceid = :faceid;')                       # "WHERE(folder = :original_folder AND file = :original_file);")
                       , {
                'new_folder':   prefix + new_folder,
                'new_file':     new_file,
                'new_identity': identity_manual,

                'faceid': faceid,
            })
            move(image, join(dirname(image), prefix + new_file))
            face_folder = join(dirname(dirname(image)),
                               identity_manual + "_faces")
            face_file = join(face_folder, original_file)[:-4] + "_0.png"
            if exists(face_file):
                move(face_file, join(face_folder,
                     prefix + new_file)[:-4] + "_0.png")
            if rename_synthesis_files:
                for e in glob.glob(f"VideoConference/VideoConference-synthesis/images/**/{original_file[:-len(extension)]}.*", recursive=True):
                    move(e, join(dirname(e), prefix + new_file))
                    db.execute(("UPDATE faces "
                                "SET folder = :new_folder, file = :new_file "
                                "WHERE(folder = :original_folder AND file = :original_file);"), {
                        'new_folder':   prefix + dirname(e),
                        'new_file':     new_file,

                        'original_folder': dirname(e),
                        'original_file': basename2(e),
                    })


def parse_string(s, isFolder=False):
    if s == 'None':
        return None
    if isFolder and s[-1] != '/':
        return s + '/'
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dbFile",
                        default="Deepfake.db",
                        help="Path to the database (.db file)")
    parser.add_argument("-i", "--input",
                        default="VideoConference/VideoConference-real/images",
                        type=str,
                        help="Input folder")
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--veryVerbose', '-vv', action='store_true')
    parser.add_argument('--manualFolder',    "-m",  dest='manualFolder', action='store_true',
                        help="The manual clustering was done in a separated `manual` folder")
    parser.add_argument('--no-manualFolder', "-nm", dest='manualFolder', action='store_false',
                        help="The manual clustering was done in place")

    args = parser.parse_args()
    DB_FILE = parse_string(args.dbFile)
    IN_FOLDER = parse_string(args.input)

    EXPECT_MANUAL_FOLDER = args.manualFolder
    if EXPECT_MANUAL_FOLDER is None:
        print("Please specify if there is an separated `manual` folder")

    else:
        verbose = args.verbose or args.veryVerbose
        very_verbose = args.veryVerbose

        conn = sqlite3.connect(DB_FILE, timeout=300)
        db = conn.cursor()
        db.execute(
            "UPDATE faces SET folder = SUBSTR(folder,3) WHERE folder LIKE './%';")

        update_with_manual_clustering(db, IN_FOLDER, EXPECT_MANUAL_FOLDER)

        conn.commit()
        conn.close()
