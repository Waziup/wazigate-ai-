FROM jomjol/raspberry-opencv

RUN [ "cross-build-start" ]

WORKDIR /

COPY wheels ./wheels

RUN apt-get update --allow-releaseinfo-change
RUN apt-get update

# tensorflow
#RUN pip3 install --no-cache-dir /wheels/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl
RUN pip3 install --no-cache-dir /wheels/tflite_runtime-2.5.0-cp37-cp37m-linux_armv7l.whl
RUN pip3 install --no-cache-dir /wheels/Pillow-6.2.0-cp37-cp37m-linux_armv7l.whl
#RUN pip3 install --no-cache-dir /wheels/numpy-1.17.4-cp37-cp37m-linux_armv7l.whl
RUN pip3 install -U numpy

# pytorch dependecies
RUN apt install libopenblas-dev libblas-dev m4 cmake cython python3-dev python3-yaml python3-setuptools #git

# ultralytics
RUN pip3 install matplotlib pandas seaborn requests tqdm

#RUN git clone https://github.com/pytorch/pytorch --recursive && cd pytorch
#RUN git checkout v1.7.0
#RUN git submodule update --init --recursive
#RUN python setup.py bdist_wheel

# torchvision
#RUN git clone https://github.com/pytorch/vision && cd vision
#RUN git checkout v0.8.1
#RUN git submodule update --init --recursive
#RUN python setup.py bdist_wheel

# pytorch & torchvision
RUN pip3 install --no-cache-dir /wheels/torch-1.7.0a0-cp37-cp37m-linux_armv7l.whl
RUN pip3 install --no-cache-dir /wheels/torchvision-0.8.0a0+45f960c-cp37-cp37m-linux_armv7l.whl


RUN pip3 install flask
RUN pip3 install flask-caching

RUN pip3 install picamera[array]

# Copy whole folder to container
COPY . /app/wazigate-webcam-inference

WORKDIR /app/wazigate-webcam-inference

# Make port available
EXPOSE 5000/tcp


# Trigger Python webcam inference script
ENTRYPOINT ["python3", "/app/wazigate-webcam-inference/app.py"]

RUN [ "cross-build-end" ]
