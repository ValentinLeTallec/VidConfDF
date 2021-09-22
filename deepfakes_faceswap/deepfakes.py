"""
Adapted from https://github.com/ondyari/FaceForensics/blob/master/dataset/DeepFakes/deepfakes.py
"""

import os
from os.path import join, expanduser, exists, basename, isdir, dirname
from os import listdir
import argparse
from tqdm import tqdm
import subprocess
import random
import shutil
import time
import datetime
import json
from distutils.dir_util import copy_tree
import re


deepfakes_path = join(dirname(__file__), "faceswap-master/faceswap.py")
verbose = True

simulate = False

def get_video_name(path: str):
    res = re.search(r"(/|^)[^/]*@\d\d:\d\d:\d\d", path)
    if res is None:
        res = re.search(r"(/|^)laptop_a\d{4}_(in|out)door", path)
    res = res.group()
    if res[0] == "/" :
        res = res[1:]
    return res

def execute(cmd: str):
    if verbose :
        print(cmd)
        print("-----------------------------")
    if not simulate: 
        # subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
        subprocess.run(cmd, shell=True) # Can be used to launch everything in parallel

def train(data_path1, data_path2, model_path, python_path, gpu=0,
          iterations=200000, nb_save = 10):
    """
        nb_save is the number of time the model should be saved during the training
        (e.g. 1 is "only save in the end", 2 is "save once in the middle of training and once in the end")
    """
    os.makedirs(model_path, exist_ok=True)
    execute('CUDA_VISIBLE_DEVICES={} {} {} train -A {} -B {} -m {} -it {:d} -s {:d}'.format(
            gpu, python_path, deepfakes_path, data_path1, data_path2,
            model_path, iterations, int(iterations/nb_save)))


def convert_frames_to_data(data_path, output_path, python_path,
                           gpu=0, alignments_path=None):
    os.makedirs(output_path, exist_ok=True)
    if alignments_path is not None:
        align = '--alignments {}'.format(alignments_path)
    else:
        align = ''
    execute('CUDA_VISIBLE_DEVICES={} {} {} extract -i {} -o {} {}'.format(
            gpu, python_path, deepfakes_path, data_path, output_path, align))


def convert(data_path, output_path, model_path, gpu, python_path,
            swap_models=False, alignments_path=None):
    os.makedirs(output_path, exist_ok=True)

    swap = '--swap-model' if swap_models else ''
    if alignments_path is not None:
        align = '--alignments {}'.format(alignments_path)
    else:
        align = ''

    execute('CUDA_VISIBLE_DEVICES={} {} {} convert -i {} -o {} -m {} {} {}'
        .format(gpu, python_path, deepfakes_path,  data_path, output_path,
                model_path, swap, align))


def generate_models(data_path, output_path, gpu, iterations, python_path,
                    filelist=None,
                    convert_images=True,
                    keep_temp_directories=True,
                    **kwargs
                    ):
    """
    Runs the full deepfakes script and creates output folders for all videos.
    We take pairs of two videos in the input path and train a model on these.

    :param data_path: contains image folders
    :param output_path: we create a single folder where we store all outputs of
    our script. This includes:
        - model files
        - manipulated images
    :param gpu: which gpu to use for training
    :param iterations: after how many iterations training has to stop
    :param python_path: absolute path to python
    :param filelist: contains pairs of input folders to manipulate. If "None" we
    take all input files and randomly
    :param convert_images: if we should convert the input images directly after
    training our models
    :param keep_temp_directories: if we should keep temporary directories
    produced by the respective sub-functions (a lot of files)
    :return:
    """
    if not filelist:
        assert len(os.listdir(data_path)) % 2 == 0,\
            'Odd number of folders in data_path, please provide a filelist ' +\
            'or delete a file'
        input_files = os.listdir(data_path)

        # Filter out what was already trained 
        # ! May filter out not fully trained models, if you try train them in 2 times !
        done_folders = os.listdir(output_path)
        done_names = []
        for folder in done_folders:
            for extract_and_identity in folder.split('⟶'):
                done_names.append(extract_and_identity)

        if len(done_names) > 0:
            print("The fellowing videos were already used and thus will be ignored : " + str(done_names)[1:-1])
        input_files = list(filter(lambda e: not (e in done_names), input_files))

        random.shuffle(input_files)
        # input_files = ["WhesSCRO-1M@00:15:15_002", "1uedfp2lF7w@00:00:35_000"]
    else:
        # Open filelist and add them all to one list, ordered pairs
        with open(filelist, 'r') as f:
            filelist = json.load(f)
            input_files = []
            for pair in filelist:
                input_files.append(pair[0])
                input_files.append(pair[1])

    print('-'*80)
    print('Starting main')
    print('-'*80)
    for i in tqdm(range(0, len(input_files), 2)):
        start_time = time.time()

        # File names for input and output
        path1 = input_files[i]
        path2 = input_files[i+1]
        output_fn = str(path1) + '⟶' + str(path2)
        tqdm.write('Starting {}'.format(output_fn))
        # Output folder
        output_folder_path = join(output_path, output_fn)
        os.makedirs(output_folder_path, exist_ok=True)

        # 1. Copy images for safety
        for apath in [path1, path2]:
            tqdm.write('Copying {} images'.format(apath))
            copy_tree(join(data_path, apath), join(output_folder_path, apath))

        # 2. Prepare images for training
        for apath in [path1, path2]:
            tqdm.write('Prepare {} images for training'.format(apath))
            convert_frames_to_data(join(output_folder_path, apath),
                                   join(output_folder_path, apath + '_faces'),
                                   gpu=gpu, python_path=python_path,
                                   alignments_path=join(output_folder_path,
                                   '{}_alignment.txt'.format(apath)))

        # Time
        prep_finished_time = time.time()
        time_taken = time.time() - start_time
        tqdm.write('Finished preparation in {}'.format(
            str(datetime.timedelta(0, time_taken))))

        # 3. Train deepfakes model
        tqdm.write('Start training with {} iterations on gpu {}'.format(
            iterations, gpu))
        train(data_path1=join(output_folder_path, path1 + '_faces'),
              data_path2=join(output_folder_path, path2 + '_faces'),
              model_path=join(output_folder_path, 'models'),
              gpu=gpu, iterations=iterations, python_path=python_path)

        # Time
        time_taken = time.time() - prep_finished_time
        tqdm.write('Finished training in {}'.format(
            str(datetime.timedelta(0, time_taken))))

        # 4. Convert images with trained model
        folders_to_keep = ['models']
        if keep_temp_directories and convert_images:
            for apath in [path1, path2]:
                tqdm.write('Converting images: {}'.format(apath))
                out_path = path1 + '⟶' + path2 if apath == path1 else \
                    path2 + '⟶' + path1
                folders_to_keep.append(out_path)
                convert(data_path=join(data_path, path1),
                        output_path=join(output_folder_path, out_path),
                        model_path=join(output_folder_path, 'models'),
                        gpu=gpu,
                        python_path=python_path,
                        alignments_path=join(output_folder_path,
                                             '{}_alignment.txt'.format(apath))
                        )

        # Cleaning up
        if not keep_temp_directories:
            tqdm.write('Cleaning up')
            for folder in os.listdir(output_folder_path):
                if folder not in folders_to_keep:
                    folder_path = join(output_folder_path, folder)
                    if os.path.isfile(folder_path):
                        os.remove(folder_path)
                    else:
                        shutil.rmtree(folder_path)

        # Time
        time_taken = time.time() - start_time
        tqdm.write('Finished in {}'.format(
            str(datetime.timedelta(0, time_taken))))

        if simulate and len(os.listdir(join(output_folder_path, 'models'))) < 1:
            os.rmdir(join(output_folder_path, 'models'))

def getIdentityFolders(folder: str):
    identityFolders = []

    get_subfolders = lambda rootFolder: [f for f in listdir(rootFolder) if isdir(join(rootFolder, f))]
    subfolders = get_subfolders(folder)

    # Only use rightly identified faces
    if any([basename(s) == "manual" for s in subfolders]):
        folder = join(folder, "manual")
        subfolders = get_subfolders(folder)

    if not "bad" in listdir(folder):
        subfolders = list(filter(lambda e: not e.endswith("_faces"), subfolders))
        for subfolder in subfolders:
            if basename(subfolder) != "tmp":
                identityFolders.append(subfolder)

    return identityFolders

def create_from_models(models_path, images_path, output_path, python_path,
                       gpu=0, **kwargs):
    images_output_path = output_path # Rename script arg to be a bit more explicit

    print('Starting process')

    count = 0
    model_folders = sorted(os.listdir(models_path))
    # model_folders = ["kt2CMf95Q0w@01:43:08_000⟶WhesSCRO-1M@00:15:15_001"]
    # model_folders = ["vekX_HwZtuM@01:45:20_001⟶xGq_ZHJ92fQ@00:33:10_000", "kt2CMf95Q0w@01:43:08_000⟶WhesSCRO-1M@00:15:15_001"]
    video_folders = sorted(os.listdir(images_path))
    print("{} | {}".format(len(model_folders),len(video_folders)))

    # 1. Extract images (for the alignment file)
    for video_folder in video_folders:
        identities_folder = join(images_path, video_folder)
        identities = sorted(getIdentityFolders(identities_folder))
        identities = [i for i in identities if not i.endswith('_faces')]
        for identity in identities:
            folder_faces = join(identities_folder, identity + '_faces')
            alignments_path = join(identities_folder, identity, '{}_{}_alignment.fsa'.format(video_folder, identity))
            if not exists(alignments_path) :
                convert_frames_to_data(join(identities_folder, identity),
                                    folder_faces,
                                    gpu=gpu, python_path=python_path,
                                    alignments_path=alignments_path)

    # 2. Convert images
    for model_folder in tqdm(model_folders):
        model_folder_reverse = '⟶'.join(model_folder.split('⟶')[::-1])
        # if not (model_folder.split('⟶')[0] in video_folders and model_folder_reverse.split('⟶')[0] in
        #         video_folders):
        #     continue
        tqdm.write('Starting {} and {}'.format(model_folder, model_folder_reverse))

        if os.path.exists(join(images_output_path, model_folder)) and \
           os.path.exists(join(images_output_path, model_folder_reverse)):
            tqdm.write('Skipping {}'.format(model_folder))
            count += 1
            continue

        for chosen_file in [model_folder, model_folder_reverse]:
            tqdm.write('Converting {}'.format(chosen_file))
            swap_models = True if chosen_file == model_folder_reverse else False
            model_path = join(models_path, model_folder)
            if os.path.exists(join(model_path, 'models')):
                model_path = join(model_path, 'models')

            extract_and_identity = chosen_file.split('⟶')[0]
            extract_name = get_video_name(extract_and_identity)
            identity     = re.search(extract_name + r"_\d{3}$", extract_and_identity).group()[-3:]

            data_path = join(images_path, extract_name, 'manual', identity)
            convert_output_path = join(images_output_path, chosen_file)
            os.makedirs(convert_output_path, exist_ok=True)

            if len(os.listdir(convert_output_path)) < 1:
                convert(data_path=data_path,
                        output_path=convert_output_path,
                        model_path=model_path,
                        gpu=gpu,
                        swap_models=swap_models,
                        alignments_path=join(data_path, extract_and_identity + '_alignment.json'),
                        python_path=python_path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    # subparsers = p.add_subparsers()
    p.add_argument('--mode', '-m', default='generate_models')
    p.add_argument('--data_path', '-i', type=str)
    p.add_argument('--data_path2', '-i2', type=str)
    p.add_argument('--output_path', '-o', type=str, default='data/out/')
    p.add_argument('--python_path', type=str, default=expanduser('~/anaconda3/bin/python3.8'))
    p.add_argument('--model_path', type=str)
    p.add_argument('--gpu', type=str, default='None')
    p.add_argument('--iterations', '-it', type=int, default=200000)
    p.add_argument('--keep_temp_directories', action='store_true')
    p.add_argument('--convert_images', action='store_true')
    p.add_argument('--filelist', type=str, default=None)
    p.add_argument('--verbose', '-v', action='store_true')
    p.add_argument("--simulate", "-s", action='store_true', help="Add this option to simulate (i.e. It won't spawn the commands)")

    args = p.parse_args()
    mode = args.mode
    simulate = args.simulate
    verbose = args.verbose or simulate

    # Convert images to input data
    if mode == 'extract_faces':
        convert_frames_to_data(data_path=args.data_path,
                               output_path=args.output_path,
                               alignments_path=join(args.output_path,
                                                    'alignments.fsa'),
                               python_path=args.python_path)
    # Train model on extracted data
    elif mode == 'train':
        train(data_path1=args.data_path, data_path2=args.data_path2,
              model_path=args.model_path, gpu=args.gpu,
              python_path=args.python_path)
    # Convert video/images with trained model
    elif mode == 'convert':
        convert(**vars(args))
    # Full script
    elif mode == 'generate_models':
        generate_models(**vars(args))
    elif mode == 'create_from_models':
        create_from_models(models_path=args.data_path,
                           images_path=args.data_path2,
                           **vars(args))

# py deepfakes_faceswap/deepfakes.py -m generate_models    -i data/src/ -o data/out/ --gpu 0
# py deepfakes_faceswap/deepfakes.py -m create_from_models -i data/out/ -i2 datasets/VideoConference/VideoConference-real/images -o datasets/VideoConference/VideoConference-synthesis/images/deepfakes_faceswap --gpu 0
