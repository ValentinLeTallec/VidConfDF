# Datasets

## Folders

* [Dataset VidConfDF](./datasets/VideoConference/README.md) : The actual dataset.
* [Datasets VidConfDF extended](./datasets/VideoConference_extended/README.md) : A version the dataset where the excerpts are longer. It is meant to be used to trained models that requiere a lot of data such as [deepfakes/faceswap](https://github.com/deepfakes/faceswap). Be aware that not all faces are well clustered in the extended dataset.

As a general rule if the folder containing the identities folders has a `good` file the clustering is right, if it contains a `bad` file the clustering is wrong or incomplete and if it contains neither then the clustering was not inspected and could be either.

## Files

* [cluster_faces.py](./cluster_faces.py) : When the default face clustering method (used while extracting the faces) does not give out satifiying results, one can use this script to try other methods.
* [face_reinsertion.py](./face_reinsertion.py) : Script used to reinsert fakes faces in their original video to create a fake video.
* [update_with_manual_clustering.py](./update_with_manual_clustering.py) : If one had to correct the automatic clustering, then they can use this script to update the database.
* [deepfake.sql](./deepfake.sql) : The SQL structure of the database. 

## Dependencies installation

```
pip install -r requierement.txt
```

## Instruction to start from scratch
If you do not wish to use the database contained in the `Deepfake.db` file from the large file archive (cf. [VidConfDF README](../README.md)), you can start from scratched with the following.

To create the database, run :
```bash
sqlite3 Deepfake.db < deepfake.sql
```

Then go to your chosen dataset's folder and launch 
```bash
python3 save-data.py
```

## Structure

```
datasets
├── VideoConference
│  ├── hcvm
│  │  ├── 000
│  │  │  ├── laptop_a0117_indoor_00000_000_001.jpg
│  │  │  ...
│  │  ...
│  │
│  ├── VideoConference-real
│  │  ├── images
│  │  │  ├── 1uedfp2lF7w@00:00:35
│  │  │  │  ├── 000
│  │  │  │  │  ├── 1uedfp2lF7w@00:00:35_000_alignment.fsa
│  │  │  │  │  ├── 1uedfp2lF7w@00:00:35_00000_000.jpg
│  │  │  │  │  ...
│  │  │  │  │  └── 1uedfp2lF7w@00:00:35_00112_000.jpg
│  │  │  │  ├── 000_faces
│  │  │  │  │  ├── 1uedfp2lF7w@00:00:35_00001_000_0.png
│  │  │  │  │  ...
│  │  │  │  ...
│  │  │  │  └── good
│  │  │  ...
│  │  │  └── xGq_ZHJ92fQ@00:33:10
│  │  │     ├── 000
│  │  │     │  ├── xGq_ZHJ92fQ@00:33:10_000_alignment.fsa
│  │  │     │  ├── xGq_ZHJ92fQ@00:33:10_00000_000.jpg
│  │  │     │  ...
│  │  │     │  └── xGq_ZHJ92fQ@00:33:10_00245_000.jpg
│  │  │     ├── 000_faces
│  │  │     │  ├── xGq_ZHJ92fQ@00:33:10_00000_000_0.png
│  │  │     │  ...
│  │  │     ...
│  │  │     └── good
│  │  └── videos
│  │     ├── 1uedfp2lF7w@00:00:35.mp4
│  │     ...
│  └── VideoConference-synthesis
│     ├── images
│     │  ├── deepfakes_faceswap
│     │  │  ├── 1uedfp2lF7w@00:00:35_000⟶WhesSCRO-1M@00:15:15_002
│     │  │  │  ├── 1uedfp2lF7w@00:00:35_00000_000.jpg
│     │  │  │  ...
│     │  │  ...
│     │  └── faceshifter
│     │     ├── 1uedfp2lF7w@00:00:35_000⟶n2Ipy5bhGQQ@00:08:12_002
│     │     │  ├── 1uedfp2lF7w@00:00:35_00042_000.jpg
│     │     │  ...
│     │     ...
│     │
│     └── videos
│        ├── deepfakes_faceswap
│        │  ├── pt-wO73Y8to@00:09:15
│        │  │  ├── pt-wO73Y8to@00:09:15-deepfakes_faceswap-0000.mp4
│        │  │  ...
│        │  ...
│        └── faceshifter
│           ├── 1uedfp2lF7w@00:00:35
│           │  ├── 1uedfp2lF7w@00:00:35_000⟶n2Ipy5bhGQQ@00:08:12_002-faceshifter.mp4
│           │  ...
│           ...
├── VideoConference_extended
│  └── VideoConference_extended-real
│     ├── images
│     │  ├── 1uedfp2lF7w@00:00:35_extended
│     │  │  ├── 000
│     │  │  │  ├── 1uedfp2lF7w@00:00:35_extended_00000_000.jpg
│     │  │  │  ...
│     │  │  ...
│     │  ...
│     └── videos
│        ├── 1uedfp2lF7w@00:00:35_extended.mp4
│        ...
└── Deepfake.db
```
For a more detailed version see [structure.md](./structure.md). 

For a more detailed version with icons for the folders, images, etc see [structure_with_icons.md](./structure_with_icons.md) (requires FontAwesome)
