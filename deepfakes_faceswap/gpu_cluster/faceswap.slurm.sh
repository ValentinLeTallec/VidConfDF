#!/bin/bash
#
#SBATCH --partition=insa-gpu
#SBATCH --job-name=df_faceswap
#SBATCH --output=/calcul-crn20/vletalle/PFE/video-conference-dataset/data/cout.txt
#SBATCH --error=/calcul-crn20/vletalle/PFE/video-conference-dataset/data/cerr.txt
#
#SBATCH -w crn20
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16000M


time srun singularity exec --bind "/calcul-crn20/vletalle/PFE:/home-reseau/vletalle/PFE" --nv deepfakes_faceswap/faceswap.sif /bin/bash -c "cd ~/PFE/video-conference-dataset && python3.7 deepfakes_faceswap/deepfakes.py -m generate_models -i data/src/ -o data/out/ --gpu 0 --python_path /usr/local/miniconda/bin/python3.7"
# time srun singularity exec --bind "/calcul-crn20/vletalle/PFE:/home-reseau/vletalle/PFE" --nv deepfakes_faceswap/faceswap.sif /bin/bash -c "cd ~/PFE/video-conference-dataset && CUDA_VISIBLE_DEVICES=0 /usr/local/miniconda/bin/python3.7 /home-reseau/vletalle/PFE/video-conference-dataset/deepfakes_faceswap/faceswap-master/faceswap.py train -A data/out/vekX_HwZtuM@01:45:20_001⟶xGq_ZHJ92fQ@00:33:10_000/vekX_HwZtuM@01:45:20_001_faces -B data/out/vekX_HwZtuM@01:45:20_001⟶xGq_ZHJ92fQ@00:33:10_000/xGq_ZHJ92fQ@00:33:10_000_faces -m data/out/vekX_HwZtuM@01:45:20_001⟶xGq_ZHJ92fQ@00:33:10_000/models -it 199975 -s 2000"
