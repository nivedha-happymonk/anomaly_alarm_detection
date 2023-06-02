import cv2
import numpy as np
import torch
import time
import simpleaudio as sa

results1_lst = []
# Load the YOLOv5 model
model = torch.hub.load('./yolov5-master', 'custom', path='three_class_05_dec.pt', source="local")


# opening the file in read mode
my_file = open("./url_file.txt", "r")
# reading the file
data = my_file.read()
# replacing end splitting the text 
# when newline ('\n') is seen.
urls = data.split("\n")
my_file.close()

def sound_alarm():
    wavfile="./alarm.wav"
    w_object=sa.WaveObject.from_wave_file(wavfile)
    p_object=w_object.play()
    p_object.wait_done()
    return

# Initialize cameras
cameras = []
for url in urls:
    if url not in [""," "]:
        cam = cv2.VideoCapture(url)
        cameras.append(cam)

def recreate_camobj(url):
    try:
        cam = cv2.VideoCapture(url)
        #cameras[i] = cam
        ret, frame = cam.read()
        #frame = np.zeros((720,1080,3), np.uint8)
        frame = cv2.resize(frame, (1080, 720))
        return cam
    except:
        recreate_camobj(url)


# Create window
cv2.namedWindow('MultiCam', cv2.WINDOW_NORMAL)
j = 0
ele=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
while True:
    # Capture frames from each camera
    j +=1
    frames = []
    for i,cam in enumerate(cameras):

        frame = []
        ret, frame = cam.read()
        try:
            frame = cv2.resize(frame, (1080, 720))
        except Exception as e:
            print(e)
            cam = recreate_camobj(urls[i])
            #cam = cv2.VideoCapture(urls[i])
            cameras[i] = cam
            ret, frame = cam.read()
            #frame = np.zeros((720,1080,3), np.uint8)
            frame = cv2.resize(frame, (1080, 720))

        frames.append(frame)
        print(frame.shape)
    # if j>20:
    # Process frames with YOLOv5
    t0 = time.time()
    processed_frames = []
    print(len(frames))
    if j % 10 == 0 or j == 1:
        results1_lst = []
        for count,frame in enumerate(frames):
            # results1_lst = []
            print("entered")
            # Detect objects using YOLOv5
            
            results1 = model(frame)
            results1_lst.append(results1)
            # Loop through YOLOv5 predictions
            for pred in results1.pred:
                # Get prediction data
                xyxy = pred[:, :4].cpu().numpy()
                conf = pred[:, 4].cpu().numpy()
                class_ids = pred[:, 5].cpu().numpy().astype(int)

                # Draw bounding boxes and labels
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    cls_id = class_ids[i]
                    cls_name = model.names[cls_id]
                    if cls_name=='Elephant':
                        ele[count]+=1
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        cv2.putText(frame, f'{cls_name}: {conf[i]:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    else:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f'{cls_name}: {conf[i]:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            processed_frames.append(frame)
        print(len(results1_lst))
    else:
        for k,frame in enumerate(frames):
            # Loop through YOLOv5 predictions
            # for results1 in results1_lst: 
            for pred in results1_lst[k].pred:
                # Get prediction data
                xyxy = pred[:, :4].cpu().numpy()
                conf = pred[:, 4].cpu().numpy()
                class_ids = pred[:, 5].cpu().numpy().astype(int)
                    
                # Draw bounding boxes and labels
                for i in range(len(xyxy)):  
                    x1, y1, x2, y2 = xyxy[i]
                    cls_id = class_ids[i]
                    cls_name = model.names[cls_id]
                    if cls_name=='Elephant':
                        ele[k]+=1
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        cv2.putText(frame, f'{cls_name}: {conf[i]:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    else:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f'{cls_name}: {conf[i]:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            processed_frames.append(frame)
         
        if any(count>10 for count in ele):
            print("\n\n**********************Alarm***********************\n\n")
            # sound_alarm()
            ele=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        print(ele)
    # print(len(processed_frames))
    t1 = time.time()
    # fps = round(1 / (t1 - t0), 2)
    rows = []
    for i in range(0, 16, 4):
    #i=0
    #if len(processed_frames) == 4:
        print(processed_frames[i].shape, processed_frames[i+1].shape ,processed_frames[i+2].shape ,processed_frames[i+3].shape)
        row = np.concatenate((processed_frames[i], processed_frames[i+1],processed_frames[i+2],processed_frames[i+3]), axis=1)
        rows.append(row)
    
    multi = np.concatenate(rows, axis=0)
    # cv2.putText(multi, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.namedWindow("MultiCam", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("MultiCam",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow('MultiCam', multi)
        
    # Check for exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and destroy the window
for cam in cameras:
    cam.release()
cv2.destroyAllWindows()