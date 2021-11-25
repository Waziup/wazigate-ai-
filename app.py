# Import object_detection
import os
#import argparse
import cv2
import numpy as np
import time
import importlib.util
from pathlib import Path

from flask import Flask, render_template, Response, request
from flask_caching import Cache
from werkzeug.serving import run_simple

import torch

from utils.plots import Annotator, colors
from utils.general import check_suffix, non_max_suppression, scale_coords
from utils.torch_utils import select_device

import logging

app = Flask(__name__)

camera = cv2.VideoCapture(0)#cv2.VideoCapture('rtsp://freja.hiof.no:1935/rtplive/_definst_/hessdalen03.stream')  # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)

# Could also add file to log
logging.basicConfig(level=logging.DEBUG, format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

MODEL_NAME = 'coco_tiny_yolov5'
GRAPH_NAME = 'detect.tflite'
LABELMAP_NAME = 'labelmap.txt'
use_TPU = False

# devalid time 
time_devalid = 10


pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# Update list of classifications and add time stamps        
def process_occurrence(occurrances,object_nr,time_occurrence):
    import time
    # If class has never been detected before, add to list
    if occurrances[object_nr][0] == 0.0:
        occurrances[object_nr][0] += 1
    # If class has been detected before, compare time_devalid
    elif time_occurrence-occurrances[object_nr][1] > 10:#time_devalid:
        #app.logger.info("Added class: after " + str(time_occurrence-occurrances[object_nr][1]) + " sec, after the last detection.", file=sys.stderr) # + labels[object_nr])
        occurrances[object_nr][0] += 1
    
    # Add last detected time to current class
    occurrances[object_nr][1] = time.time()
        
    return occurrances

def get_labels(PATH_TO_LABELS):
    # Load the label map
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels


def gen_frames_mobile(MODEL_NAME):
    import time

    cache = Cache(app, config={
        'CACHE_TYPE': 'filesystem',
        'CACHE_DIR': 'my_cache_directory',
        'CACHE_DEFAULT_TIMEOUT': 86400, # keep cache for 24 hours
    })

    MODEL_NAME = 'coco_ssd_mobilenet_v1'
    GRAPH_NAME = 'detect.tflite'
    LABELMAP_NAME = 'labelmap.txt'
    
    min_conf_threshold = 0.59
    resW = '640'
    resH = '480'
    imW = int(resW)
    imH = int(resH)
    use_TPU = False
    
    # If using Edge TPU, assign filename for Edge TPU model
    if use_TPU:
        # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
        if (GRAPH_NAME == 'detect.tflite'):
            GRAPH_NAME = 'edgetpu.tflite'       
    
    # Get path to current working directory
    CWD_PATH = os.getcwd()
    
    # Path to .tflite file, which contains the model that is used for object detection
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)
    
    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)
    
    labels = get_labels(PATH_TO_LABELS)
    
    # Have to do a weird fix for label map if using the COCO "starter model" from
    # https://www.tensorflow.org/lite/models/object_detection/overview
    # First label is '???', which has to be removed.
    if labels[0] == '???':
        del(labels[0])
        
    # Count occurances and time init ndarry with zeros
    s = (len(labels),2)
    occurrances = np.zeros(s)
    cache.set("labels", labels)
    
    # Load the Tensorflow Lite model.
    # If using Edge TPU, use special load_delegate argument
    if use_TPU:
        interpreter = Interpreter(model_path=PATH_TO_CKPT,
                                  experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
        print(PATH_TO_CKPT)
    else:
        interpreter = Interpreter(model_path=PATH_TO_CKPT)
    
    interpreter.allocate_tensors()
    
    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    
    floating_model = (input_details[0]['dtype'] == np.float32)
    
    input_mean = 127.5
    input_std = 127.5
    
    # Initialize frame rate calculation
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()
    
    #time.sleep(1)
    
    
    while True:
        
        # Capture frame-by-frame
        success, frame1 = camera.read()  # read the camera frame
        if not success:
            print("camera.read() failed")
            break
        else:
             # Start timer (for calculating frame rate)
            t1 = cv2.getTickCount()
        
            # Acquire frame and resize to expected shape [1xHxWx3]
            frame = frame1.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (width, height))
            input_data = np.expand_dims(frame_resized, axis=0)
        
            # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
            if floating_model:
                input_data = (np.float32(input_data) - input_mean) / input_std
        
            # Perform the actual detection by running the model with the image as input
            interpreter.set_tensor(input_details[0]['index'],input_data)
            interpreter.invoke()
        
            # Retrieve detection results
            boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
            classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
            scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
            #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
        
            # Loop over all detections and draw detection box if confidence is above minimum threshold
            for i in range(len(scores)):
                if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1,(boxes[i][0] * imH)))
                    xmin = int(max(1,(boxes[i][1] * imW)))
                    ymax = int(min(imH,(boxes[i][2] * imH)))
                    xmax = int(min(imW,(boxes[i][3] * imW)))
                    # Draw box
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                    # Draw label
                    object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                    
                    # Evaluate occurrances               
                    occurrances = process_occurrence(occurrances,int(classes[i]),time.time())
           
            # Calculate framerate
            t2 = cv2.getTickCount()
            time1 = (t2-t1)/freq
            frame_rate_calc= 1/time1
           
            # Draw framerate in corner of frame
            cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(0,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
           
            ret, buffer = cv2.imencode('.jpg', frame)
            app.logger.info("MobileNet: Shape of result image: " + str(frame.shape))
            
            frame = buffer.tobytes()
                    # Store occurrances in cache to be able to retriev information later
            cache.set("occurrances", occurrances)
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
            
            
def gen_frames_yolo(MODEL_NAME):
    # for yolov5
    conf_thres=0.25  # confidence threshold
    iou_thres=0.45  # NMS IOU threshold
    max_det=1000  # maximum detections per image
    agnostic_nms=False  # class-agnostic NMS
    line_thickness=3  # bounding box thickness (pixels)
    classes=None  # filter by class: --class 0, or --class 0 2 3
    imgsz=320   # inference size (pixels)
    GRAPH_NAME = 'detect.tflite'
    
    # prepare cache
    cache = Cache(app, config={
        'CACHE_TYPE': 'filesystem',
        'CACHE_DIR': 'my_cache_directory',
        'CACHE_DEFAULT_TIMEOUT': 86400, # keep cache for 24 hours
    })

    app.logger.info("yolov5 : " + MODEL_NAME + GRAPH_NAME)


    # Initialize
    device='cpu'
    half = False
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    suffix, suffixes = Path(GRAPH_NAME).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
    check_suffix(GRAPH_NAME, suffixes)  # check weights have acceptable suffix
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
    names = [f'class{i}' for i in range(1000)]  # assign defaults
    
    # Get path to current working directory
    CWD_PATH = os.getcwd()
    
    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)
    
    labels = get_labels(PATH_TO_LABELS)
    
    # Count occurances and time init ndarry with zeros
    s = (len(labels),2)
    occurrances = np.zeros(s)
    cache.set("labels", labels)
            
    interpreter = Interpreter(model_path=MODEL_NAME + '/' + GRAPH_NAME)  # load TFLite model
    interpreter.allocate_tensors()  # allocate
    input_details = interpreter.get_input_details()  # inputs
    output_details = interpreter.get_output_details()  # outputs
    int8 = input_details[0]['dtype'] == np.uint8  # is TFLite quantized uint8 model
    half = input_details[0]['dtype'] == np.float16 # is TFLite is float16 model
    imgsz = (input_details[0]['shape'][1],input_details[0]['shape'][2])
    #imgsz = check_img_size(imgsz, s=stride)  # check image size


    # Initialize frame rate calculation
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    #for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
    # Run inference
    seen = 0
    while(True):
        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()
        
        success, frame1 = camera.read()  # read the camera frame
        img = cv2.resize(frame1, imgsz, interpolation = cv2.INTER_AREA)
        img = img[None]
        img = np.transpose(img, (0,3,1,2))
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        # Inference
        imn = img.permute(0, 2, 3, 1).cpu().numpy()  # image in numpy
        if int8:
            scale, zero_point = input_details[0]['quantization']
            imn = (imn / scale + zero_point).astype(np.uint8)  # de-scale
        interpreter.set_tensor(input_details[0]['index'], imn)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])
        if int8:
            scale, zero_point = output_details[0]['quantization']
            pred = (pred.astype(np.float32) - zero_point) * scale  # re-scale
        pred[..., 0] *= imgsz[1]  # x
        pred[..., 1] *= imgsz[0]  # y
        pred[..., 2] *= imgsz[1]  # w
        pred[..., 3] *= imgsz[0]  # h
        pred = torch.tensor(pred)
        
        print("YOLOv5: Prediction of result image: " + str(pred.shape))
        
        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)


        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            
            s, im0 = f'{i}: ', frame1.copy()
    

            s += '%gx%g ' % img.shape[2:]  # print string
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
    
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = f'{labels[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    
                    # Evaluate occurrances               
                    occurrances = process_occurrence(occurrances,c,time.time())
            
            # Calculate framerate
            t2 = cv2.getTickCount()
            time1 = (t2-t1)/freq
            frame_rate_calc= 1/time1

            # Stream results
            im0 = annotator.result()
            app.logger.info("YOLOv5: Shape of result image: " + str(im0.shape))
            
            cache.set("occurrances", occurrances)
            
            # Draw framerate in corner of frame
            cv2.putText(im0,'FPS: {0:.2f}'.format(frame_rate_calc),(0,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
            
            ret, buffer = cv2.imencode('.jpg', im0)
            
            im0 = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + im0 + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    model = request.args.get('model')
    app.logger.info("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX_______Modelname: " + MODEL_NAME)
    
    if model == 'coco_ssd_mobilenet_v1':
        return Response(gen_frames_mobile(model), mimetype='multipart/x-mixed-replace; boundary=frame')
    elif model == 'coco_tiny_yolov5':
        return Response(gen_frames_yolo(model), mimetype='multipart/x-mixed-replace; boundary=frame')
    

@app.route('/', methods=['GET', 'POST'])
def index():
    """Video streaming home page."""
    data={
    	'model': request.args.get('model')
    }
    
    return render_template('index.html', data=data)


@app.route("/results")
def results():
    """Show results of inference"""
    from time import sleep
    
    cache = Cache(app, config={
        'CACHE_TYPE': 'filesystem',
        'CACHE_DIR': 'my_cache_directory',
        'CACHE_DEFAULT_TIMEOUT': 86400, # keep cache for 24 hours
    })
    
    def makeTableHTML(myArray,labels_list):
        rows = np.size(myArray,0)
        row_switched = False
        result = "<div class ='center'> <div class='row'> <div class='column'> <table> <tr><th>Class</th><th>Occurrances</th><th>Timestamp last occurrance</th></tr>"
        for i in range(0,rows):
            result += "<tr>"
            for j in range(0,np.size(myArray,1)+1):
                if myArray[i][j-1] != 0:
                    if j == 0:
                        result += "<th>"+labels_list[i]+"</th>"
                    elif j == 1:
                        result += "<td>"+str(int(myArray[i][j-1]))+"</td>"
                    elif j == 2:
                        if myArray[i][j-1] == 0.0:
                            result += "<td> - </td>"   
                        else:
                            result += "<td>"+str(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(myArray[i][j-1])))+"</td>"
            if i >= rows/2 and not row_switched:
                row_switched = True
                result += "</tr>"
                result += "</table>"
                result += "</div>"
                result += "<div class='column'> <table> <tr><th>Class</th><th>Occurrances</th><th>Timestamp last occurrance</th></tr>"
            result += "</tr>"
        result += "</table>"
        result += "</div>"
        result += "</div>"
        result += "</div>"
    
        return result
    
    def streamer():
        while True:
            # Get a cached value
            occurrances = cache.get("occurrances")
            labels = cache.get("labels")
            #app.logger.info("CACHE TEST: " + str(occurrances))
            
            return "<p>{}</p>".format(makeTableHTML(occurrances,labels))
            sleep(1)

    return Response(streamer())


def main():  
    # Run local:
    app.run(host='0.0.0.0',use_reloader=False,debug=True)
    #app.run(host='unix:///var/lib/waziapp/proxy.sock',use_reloader=False,debug=True)
    
    # Run as waziapp via sockets
    #   run_simple('unix:///var/lib/waziapp/proxy.sock', 0, app, threaded=True)
    

if __name__ == '__main__':
    main()
