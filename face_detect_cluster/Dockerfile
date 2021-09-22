# set base image (host OS)
FROM nvidia/cuda:10.1-cudnn7-runtime

# set the working directory in the container
WORKDIR /app

RUN apt-get update --fix-missing
RUN apt-get install \ 
    build-essential \
    python3.8 \
    python3-pip \
    ffmpeg libsm6 \
    libxext6 \
    cmake \
    libopenblas-dev -y

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip3 install --upgrade pip 
RUN pip3 install --no-cache-dir --user -r requirements.txt

# copy the content of the local src directory to the working directory
COPY face_detect.py .
COPY face_detect_hcvm.py .
COPY insightface ./insightface
COPY models ./models

RUN cd insightface/detection/RetinaFace/rcnn/cython/ && \
    python3 setup.py build_ext --inplace && \
    rm -rf build

# command to run on container start
# ENTRYPOINT [ "python3", "face_detect_hcvm.py", "-s", "40"] 
ENTRYPOINT [ "python3", "face_detect.py", "-s", "40"]
#, "-n", "300"] 
