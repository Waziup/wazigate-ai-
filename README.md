# WaziGate-Webcam-Inference

Wazigate-Webcam-Inference is an application for the WaziGate that implements computer vision models for object detection.  

## Description

WaziApp containing object detection example for various models.With the help of this application you can utilize the WaziGate and RaspiCam to track objects and evaluate their occurrences.

The models are trained with the COCO (Common Objects in Context) [1] dataset. It contains 80 classes, you can have a look on them [here](coco_tiny_yolov5/labelmap.txt "labelmap.txt"). 

From now you can choose between two models for object detection.

1. Mobile Net v1 (320x320)
2. Yolo Version 5 (320x320) [3]

The App is in an early stage of development, it is still work in progress, some features are missing.

### Installation

You can just download the build docker container from Dockerhub.
If you want to build it from source you have to follow these steps.

1. Download the repository to your local machine:

```
git clone https://github.com/Waziup/wazigate-webcam-inference.git
```

2. Install docker and issue the following commands

```
docker buildx create --name rpibuilder --platform linux/arm/v7
docker buildx use rpibuilder 
docker buildx inspect –bootstrap
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
```

3. Navigate to the repository:

```
cd wazigate-webcam-inference
```

4. Issue the following command to build the docker image from the [Dockerfile](Dockerfile "Dockerfile"):

```
docker buildx build –platform linux/arm/v7 -t waziup/wazigate-webcam-inference:latest –load .
```

5. To copy the image via SSH to the raspberry pi with the following command:

```
docker save id_of_build_image | gzip | pv | ssh pi@ip_of_pi_in_local_network docker load
```

6. It can occur that the name of the repository is lost, so tag the image appropriate to the docker-compose.yml

```
docker tag id_of_build_image waziup/wazigate-webcam-inference:latest
```

7. Afterwards, start the application with via the UI of the WaziGate or run the following command, to see the logs in console:

```
docker-compose up
```


[1] https://cocodataset.org/#home

[3] https://github.com/ultralytics/yolov5
