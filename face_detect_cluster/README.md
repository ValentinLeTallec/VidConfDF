# Detect an cluster faces

For detection, we use RetinaFace from InsightFace (https://github.com/deepinsight/insightface/tree/0cc88e2a2449973e77dbdd417c9ec97ef03940b9/detection/RetinaFace).

For clustering, we use the python package `face_recognition`, mean-shift and sometime, when everything fails, the good old `do it by hand`.

## Set up
1. If you did not merge the the VidConfDF directory with the one containing all large files (cf. [VidConfDF README](../README.md)) you can also download a pre-trained RetinaFace [here](https://github.com/deepinsight/insightface/tree/master/detection/retinaface#retinaface-pretrained-models) 


## Build the Dockerfile :
`cd path/to/VidConfDF`

`docker build face_detect_cluster --tag=retina`

## Run the docker container
`docker rm retina_c; docker run --gpus all -v "/path/VidConfDF/datasets/:/app/data" --name retina_c retina`
