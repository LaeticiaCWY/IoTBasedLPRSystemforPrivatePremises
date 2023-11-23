import cv2

import urllib.parse
from local_utils import detect_lp
from os.path import splitext, basename
from keras.models import model_from_json
import glob
import time
from paddleocr import PaddleOCR
import argparse
import sys
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import mysql.connector
from mysql.connector import Error
import pyrebase
import firebase_admin
from firebase_admin import credentials, messaging
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, is_ascii, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, load_classifier, time_sync
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
plt.ion() 
from datetime import datetime

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

config = {
    "apiKey": "AIzaSyAZfHF22hO29tzwfGbh8mroC69Z7Qbfk1o",
    "authDomain": "lpr1-2a779.firebaseapp.com",
    "databaseURL": "https://lpr1-2a779-default-rtdb.asia-southeast1.firebasedatabase.app",
    "projectId": "lpr1-2a779",
    "storageBucket": "lpr1-2a779.appspot.com",
    "messagingSenderId": "759846216195",
    "appId": "1:759846216195:web:c69c70bf769766b2b9e436",
    "measurementId": "G-F283Z4ZW7T",
    "serviceAccount": r'C:\Users\Laeticia\Vehicle-Detection\firebase-credentials.json'
}
firebase = pyrebase.initialize_app(config)
storage = firebase.storage()
db = firebase.database()



@torch.no_grad()
def run(weights='yolov5s.pt',  # model.pt path(s)
        source='0',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='vehicle_images',  # save results to project/name
        name='test',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        timer_duration=5):
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(project) / name
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    classify, suffix = False, Path(w).suffix.lower()
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in ['.pt', '.onnx', '.tflite', '.pb', ''])  # backend
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    elif onnx:
        check_requirements(('onnx', 'onnxruntime'))
        import onnxruntime
        session = onnxruntime.InferenceSession(w, None)
    else:  # TensorFlow models
        check_requirements(('tensorflow>=2.4.1',))
        import tensorflow as tf
        if pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            def wrap_frozen_graph(gd, inputs, outputs):
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped import
                return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                               tf.nest.map_structure(x.graph.as_graph_element, outputs))

            graph_def = tf.Graph().as_graph_def()
            graph_def.ParseFromString(open(w, 'rb').read())
            frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")
        elif saved_model:
            model = tf.keras.models.load_model(w)
        elif tflite:
            interpreter = tf.lite.Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            int8 = input_details[0]['dtype'] == np.uint8  # is TFLite quantized uint8 model
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    ascii = is_ascii(names)  # names are ascii (use PIL for UTF-8)

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    cv2.destroyAllWindows()

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    # Add timer start time
    timer_start = t0

    for path, img, im0s, vid_cap in dataset:
        if onnx:
            img = img.astype('float32')
        else:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
    	

        # Inference
        t1 = time_sync()
        if pt:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(img, augment=augment, visualize=visualize)[0]
        elif onnx:
            pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))
        else:  # tensorflow model (tflite, pb, saved_model)
            imn = img.permute(0, 2, 3, 1).cpu().numpy()  # image in numpy
            if pb:
                pred = frozen_func(x=tf.constant(imn)).numpy()
            elif saved_model:
                pred = model(imn, training=False).numpy()
            elif tflite:
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

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_sync()

        # Add timer check
        if time.time() - timer_start > timer_duration:
            break  # Stop loop after timer_duration

        # Second-stage classifier (optional)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # detections per image
           
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, pil=not ascii)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{' ' * (n > 1)}, "  # add to string

                # Write results

                for *xyxy, conf, cls in reversed(det):

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)


     
            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)


        if len(det):
           print(f"Number of detections: {len(det)}")  # Debugging print statement
           for idx, (*xyxy, conf, cls) in enumerate(det):
               print(f"Saving vehicle image: {p.stem}_vehicle_{idx}.jpg")
               vehicle_img = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
               vehicles_dir = save_dir / 'images'  # Save in the 'vehicle_images' folder
               vehicles_dir.mkdir(parents=True, exist_ok=True)  # Create 'vehicle_images' directory if not exists
               vehicle_save_path = str(vehicles_dir / f'{p.stem}_vehicle_{idx}.jpg')  # Using idx for unique filenames
               cv2.imwrite(vehicle_save_path, vehicle_img)
               print(f"Saved vehicle image: {vehicle_save_path}")


  

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    print(f'Done. ({time.time() - t0:.3f}s)')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='vehicle_images', help='save results to project/name')
    # parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--timer-duration', type=int, default=5, help='duration in seconds to run detection')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt



opt = parse_opt()
print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
check_requirements(exclude=('tensorboard', 'thop'))
run(**vars(opt))
cv2.destroyAllWindows()


# Storage path for the image
storage_path1 = "lpr/YOLO.jpg"

# Local file path to upload
local_file_path1 = r'C:\Users\Laeticia\Vehicle-Detection\vehicle_images\test\images\1_vehicle_0.jpg'

# Upload the image to Firebase Storage
storage.child(storage_path1).put(local_file_path1)

# Database path for the image URL
db_path1 = "/lpr/YOLO_url"

# Customize metadata as needed (e.g., cacheControl)
metadata = {'cacheControl': 'no-cache'}

# Get the image URL from Firebase Storage
image_url = storage.child(storage_path1).get_url(None)

# Add a timestamp as a query parameter to the image URL
timestamp = int(datetime.timestamp(datetime.now()))  # Current timestamp
parsed_url = urllib.parse.urlparse(image_url)
query_params = urllib.parse.parse_qs(parsed_url.query)
query_params['timestamp'] = [str(timestamp)]
new_query = urllib.parse.urlencode(query_params, doseq=True)
updated_image_url = urllib.parse.urlunparse((
    parsed_url.scheme,
    parsed_url.netloc,
    parsed_url.path,
    parsed_url.params,
    new_query,
    parsed_url.fragment
))

# Set the image URL with timestamp to Firebase Realtime Database
db.child(db_path1).set(updated_image_url)

# Print the updated image URL
print("Updated Image URL with Timestamp:", updated_image_url)

start_time = time.time()
def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Loading model successfully...")
        return model
    except Exception as e:
        print(e)
wpod_net_path = "wpod-net.json"
wpod_net = load_model(wpod_net_path)

def preprocess_image(image_path,resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img
# Create a list of image paths 
subfolder_name = "test"
subfolder2_name = "images"

# Construct the path to the subfolder
subfolder_path = f"vehicle_images/{subfolder_name}/{subfolder2_name}/*.jpg"

# Get a list of image paths from the subfolder
image_paths = glob.glob(subfolder_path)

print("Found %i images..."%(len(image_paths)))


# forward image through model and return plate's image and coordinates
# if error "No Licensese plate is founded!" pop up, try to adjust Dmin
def get_plate(image_path, Dmax=608, Dmin=300):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return LpImg, cor

# Obtain plate image and its coordinates from an image
test_image = image_paths[0]


LpImg, cor = get_plate(test_image)


print("Detect %i plate(s) in" % len(LpImg), splitext(basename(test_image))[0])
print("Coordinate of plate(s) in image: \n", cor)



def draw_box(image_path, cor, thickness=3): 
    pts=[]  
    x_coordinates=cor[0][0]
    y_coordinates=cor[0][1]
    # store the top-left, top-right, bottom-left, bottom-right 
    # of the plate license respectively
    for i in range(4):
        pts.append([int(x_coordinates[i]),int(y_coordinates[i])])
    
    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1,1,2))
    vehicle_image = preprocess_image(image_path)
    
    cv2.polylines(vehicle_image,[pts],True,(0,255,0),thickness)
    return vehicle_image


# Save the detected license plate image
license_plate_image_path = r'C:\Users\Laeticia\Vehicle-Detection\sample_license_plate.jpg'
plt.figure(figsize=(5, 5))
plt.axis(False)


plt.imshow(LpImg[0])
# plt.title("Detected License Plate")
end_time = time.time()
elapsed_time = end_time - start_time
print("Time taken to process image: %.4f seconds" % elapsed_time)
plt.tight_layout(True)
plt.savefig(license_plate_image_path)
plt.ion()
# plt.show(block=True)

# Storage path for the image
storage_path2 = "lpr/WPOD-NET.jpg"

# Local file path to upload
local_file_path2 = r'C:\Users\Laeticia\Vehicle-Detection\sample_license_plate.jpg'

# Upload the image to Firebase Storage
storage.child(storage_path2).put(local_file_path2)

# Database path for the image URL
db_path2 = "/lpr/WPOD-NET_url"

# Customize metadata as needed (e.g., cacheControl)
metadata = {'cacheControl': 'no-cache'}

# Get the image URL from Firebase Storage
image_url = storage.child(storage_path2).get_url(None)

# Add a timestamp as a query parameter to the image URL
timestamp = int(datetime.timestamp(datetime.now()))  # Current timestamp
parsed_url = urllib.parse.urlparse(image_url)
query_params = urllib.parse.parse_qs(parsed_url.query)
query_params['timestamp'] = [str(timestamp)]
new_query = urllib.parse.urlencode(query_params, doseq=True)
updated_image_url = urllib.parse.urlunparse((
    parsed_url.scheme,
    parsed_url.netloc,
    parsed_url.path,
    parsed_url.params,
    new_query,
    parsed_url.fragment
))

# Set the image URL with timestamp to Firebase Realtime Database
db.child(db_path2).set(updated_image_url)

# Print the updated image URL
print("Updated Image URL with Timestamp:", updated_image_url)


def check_plate_number(word):
    try:
        # Connect to the database
        connection = mysql.connector.connect(
            host="sql12.freemysqlhosting.net",
            user="sql12654624",
            passwd="2GTpcCbLXC",
            database="sql12654624"  # Replace with your database name
        )

        # Create a cursor
        cursor = connection.cursor()

        # Define the SQL query to check if the plate number exists
        select_query = "SELECT Vehicle_No_Plate FROM LPRDATABASE WHERE Vehicle_No_Plate = %s"

        # Execute the query with the plate number as a parameter
        cursor.execute(select_query, (word,))

        # Fetch the result
        result = cursor.fetchone()

        # Close the cursor and connection
        cursor.close()
        connection.close()

        if result:
            print("Recognized")
            text = "Recognized"
            file_path = r'C:\Users\Laeticia\Vehicle-Detection\Recognized.txt'
            with open(file_path, "w") as text_file:
                 text_file.write(text)

             # Storage path for the image
            storage_path3 = "lpr/Recognized.txt"

             # Local file path to upload
            local_file_path3 = r'C:\Users\Laeticia\Vehicle-Detection\Recognized.txt'

             # Upload the image to Firebase Storage
            storage.child(storage_path3).put(local_file_path3)

             # Database path for the image URL
            db_path3 = "/lpr/Recognized_url"

             # Customize metadata as needed (e.g., cacheControl)
            metadata = {'cacheControl': 'no-cache'}

             # Get the image URL from Firebase Storage
            txt_url = storage.child(storage_path3).get_url(None)

             # Add a timestamp as a query parameter to the image URL
            timestamp = int(datetime.timestamp(datetime.now()))  # Current timestamp
            parsed_url = urllib.parse.urlparse(txt_url)
            query_params = urllib.parse.parse_qs(parsed_url.query)
            query_params['timestamp'] = [str(timestamp)]
            new_query = urllib.parse.urlencode(query_params, doseq=True)
            updated_txt_url = urllib.parse.urlunparse((
               parsed_url.scheme,
               parsed_url.netloc,
               parsed_url.path,
               parsed_url.params,
               new_query,
               parsed_url.fragment
               ))

            # Set the image URL with timestamp to Firebase Realtime Database
            db.child(db_path3).set(updated_txt_url)

            # Print the updated image URL
            print("Updated Txt URL with Timestamp:", updated_txt_url)
            cred = credentials.Certificate(r'C:\Users\Laeticia\react-chat\firebase-credentials.json')
            firebase_admin.initialize_app(cred)
            message = messaging.Message(
                notification=messaging.Notification(
                    title= f"License Plate {word} Recognized",
                    body= "Have a great day",
                ),
                token="dzR5ZgEu43QtvcrHfxRfBh:APA91bGR90f7emYJawvs_dewRgFijyPyjrFHfA7_Q9paPiVRLrGJannofoERWqAkXyetc4x6UBHMChCsQkx16iKIm4kkWmSQybOGBGUEKVzve4OGVQx3AyuYTHyeDeWPxy0S2oyG7zA6",
)

            response = messaging.send(message)
            print("Successfully sent message:", response)

        else:
            print("Not Recognized")
            text = "Not Recognized" 
            file_path = r'C:\Users\Laeticia\Vehicle-Detection\Recognized.txt'
            with open(file_path, "w") as text_file:
                 text_file.write(text)

            # Storage path for the image
            storage_path3 = "lpr/Recognized.txt"

            # Local file path to upload
            local_file_path3 = r'C:\Users\Laeticia\Vehicle-Detection\Recognized.txt'

            # Upload the image to Firebase Storage
            storage.child(storage_path3).put(local_file_path3)

            # Database path for the image URL
            db_path3 = "/lpr/Recognized_url"

            # Customize metadata as needed (e.g., cacheControl)
            metadata = {'cacheControl': 'no-cache'}

            # Get the image URL from Firebase Storage
            txt_url = storage.child(storage_path3).get_url(None)

            # Add a timestamp as a query parameter to the image URL
            timestamp = int(datetime.timestamp(datetime.now()))  # Current timestamp
            parsed_url = urllib.parse.urlparse(txt_url)
            query_params = urllib.parse.parse_qs(parsed_url.query)
            query_params['timestamp'] = [str(timestamp)]
            new_query = urllib.parse.urlencode(query_params, doseq=True)
            updated_txt_url = urllib.parse.urlunparse((
               parsed_url.scheme,
               parsed_url.netloc,
               parsed_url.path,
               parsed_url.params,
               new_query,
               parsed_url.fragment
            ))

           # Set the image URL with timestamp to Firebase Realtime Database
            db.child(db_path3).set(updated_txt_url)

           # Print the updated image URL
            print("Updated Txt URL with Timestamp:", updated_txt_url)
            cred = credentials.Certificate(r'C:\Users\Laeticia\react-chat\firebase-credentials.json')
            firebase_admin.initialize_app(cred)


            message = messaging.Message(
                notification=messaging.Notification(
                    title= "License Plate Not Recognized",
                    body= "Please log in to ChatApp to Assist Driver",
                ),
                token="dRfrxalSUDQ7OLqdp2kZnN:APA91bHX73eWr5Ygp6ZT1vnpy7rxBxFgjRWbtT1Zb2pUyAzpq1thXerR0X-7PRQx4liQdPcPJNh-IiO-WShSHjG9e-uIUsmRQO_cMaRo8pe-ELKJGS9nwbQUYQmoZvfKnHlWRH6-Kghi",
)

            response = messaging.send(message)
            print("Successfully sent message:", response)

            message = messaging.Message(
                notification=messaging.Notification(
                    title= "License Plate Not Recognized",
                    body= "Please log in to ChatApp to Assist Driver",
                ),
                token="ekrX4SVGLTtkkGUl-RC6bb:APA91bFJVApWXJcVZ_mkTy4Vvbm4bhJzGfH6meSTQDH-j3Y4fmd_4tzt4_ZHgr5XniuSzkSWafTj488e4HMpUJFDoeOp5ph1XDj_5AFxo1IffBUCFHXVX4QyGYUZzTCZsu4vlNHh5-MI",
)

            response = messaging.send(message)
            print("Successfully sent message:", response)

            response = messaging.send(message)
            print("Successfully sent message:", response)

            message = messaging.Message(
                notification=messaging.Notification(
                    title= f"License Plate {word} Not Recognized",
                    body= "Please Wait for Security Personnel to Assist You",
                ),
                token="dzR5ZgEu43QtvcrHfxRfBh:APA91bGR90f7emYJawvs_dewRgFijyPyjrFHfA7_Q9paPiVRLrGJannofoERWqAkXyetc4x6UBHMChCsQkx16iKIm4kkWmSQybOGBGUEKVzve4OGVQx3AyuYTHyeDeWPxy0S2oyG7zA6",
)

            response = messaging.send(message)
            print("Successfully sent message:", response)


    except Error as e:
        print(f"Error: {e}")

from paddleocr import PaddleOCR,draw_ocr
# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `french`, `german`, `korean`, `japan`
# to switch the language model in order.
ocr = PaddleOCR(use_angle_cls=True, lang='en',show_log = False) # need to run only once to download and load model into memory
result = ocr.ocr(license_plate_image_path, cls=True)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)

result = ocr.ocr(license_plate_image_path, cls=False)


def print_all_words(result):
    for sublist in result:
        if isinstance(sublist, list):
            print_all_words(sublist)
        elif isinstance(sublist, tuple):
            word = sublist[0]
            word_type = type(word)  # Get the type of the word
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"Word: {word}, Current Time: {current_time}")
            file_path = r'C:\Users\Laeticia\Vehicle-Detection\OCR.txt'
            with open(file_path, "w") as text_file:
                text_file.write(word + " , ")
                text_file.write(current_time)
            check_plate_number(word)  # Call check_plate_number with the word

print_all_words(result)


# Storage path for the image
storage_path3 = "lpr/OCR.txt"

# Local file path to upload
local_file_path3 = r'C:\Users\Laeticia\Vehicle-Detection\OCR.txt'

# Upload the image to Firebase Storage
storage.child(storage_path3).put(local_file_path3)

# Database path for the image URL
db_path3 = "/lpr/OCR_url"

# Customize metadata as needed (e.g., cacheControl)
metadata = {'cacheControl': 'no-cache'}

# Get the image URL from Firebase Storage
txt_url = storage.child(storage_path3).get_url(None)

# Add a timestamp as a query parameter to the image URL
timestamp = int(datetime.timestamp(datetime.now()))  # Current timestamp
parsed_url = urllib.parse.urlparse(txt_url)
query_params = urllib.parse.parse_qs(parsed_url.query)
query_params['timestamp'] = [str(timestamp)]
new_query = urllib.parse.urlencode(query_params, doseq=True)
updated_txt_url = urllib.parse.urlunparse((
    parsed_url.scheme,
    parsed_url.netloc,
    parsed_url.path,
    parsed_url.params,
    new_query,
    parsed_url.fragment
))

# Set the image URL with timestamp to Firebase Realtime Database
db.child(db_path3).set(updated_txt_url)

# Print the updated image URL
print("Updated Txt URL with Timestamp:", updated_txt_url)








