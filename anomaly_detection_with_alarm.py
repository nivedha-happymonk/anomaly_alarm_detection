import cv2
import numpy as np
import torch
import time
import simpleaudio as sa

def sound_alarm():
    wavfile = "./alarm.wav"
    w_object = sa.WaveObject.from_wave_file(wavfile)
    p_object = w_object.play()
    #p_object.wait_done()
    
def recreate_camobj(url):
    try:
        cam = cv2.VideoCapture(url)
        ret, frame = cam.read()
        frame = cv2.resize(frame, (1080, 720))
        return cam
    except:
        return recreate_camobj(url)
    
def main():
    # Load the YOLOv5 model
    model = torch.hub.load('./yolov5-master', 'custom', path='three_class_05_dec.pt', source="local")
    # Read URLs from the file
    with open("./url_file.txt", "r") as my_file:
        urls = my_file.read().split("\n")
    # Initialize cameras
    cameras = []
    for url in urls:
        if url not in ["", " "]:
            cam = cv2.VideoCapture(url)
            cameras.append(cam)
    # Create window
    cv2.namedWindow('MultiCam', cv2.WINDOW_NORMAL)
    j = 0
    ele = {}
    while True:
        # Capture frames from each camera
        j += 1
        frames = []
        for i, cam in enumerate(cameras):
            frame = []
            ret, frame = cam.read()
            try:
                frame = cv2.resize(frame, (1080, 720))
            except Exception as e:
                print(e)
                cam = recreate_camobj(urls[i])
                cameras[i] = cam
                ret, frame = cam.read()
                frame = cv2.resize(frame, (1080, 720))
            frames.append(frame)
            print(frame.shape)
        # Process frames with YOLOv5
        t0 = time.time()
        processed_frames = []
        if j % 10 == 0 or j == 1:
            results1_lst = []
            for count, frame in enumerate(frames):
                results1 = model(frame)
                results1_lst.append(results1)
                for pred in results1.pred:
                    xyxy = pred[:, :4].cpu().numpy()
                    conf = pred[:, 4].cpu().numpy()
                    class_ids = pred[:, 5].cpu().numpy().astype(int)
                    for i in range(len(xyxy)):
                        x1, y1, x2, y2 = xyxy[i]
                        cls_id = class_ids[i]
                        cls_name = model.names[cls_id]
                        if cls_name == 'Elephant':
                            ele.setdefault(count, 0)
                            ele[count] += 1
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                            cv2.putText(frame, f'{cls_name}: {conf[i]:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        else:
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(frame, f'{cls_name}: {conf[i]:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                processed_frames.append(frame)
            if any(count > 10 for count in ele.values()):
                print("\n\n**********************Alarm***********************\n\n")
                sound_alarm()
                ele = {}
        else:
            for k, frame in enumerate(frames):
                for pred in results1_lst[k].pred:
                    xyxy = pred[:, :4].cpu().numpy()
                    conf = pred[:, 4].cpu().numpy()
                    class_ids = pred[:, 5].cpu().numpy().astype(int)
                    for i in range(len(xyxy)):
                        x1, y1, x2, y2 = xyxy[i]
                        cls_id = class_ids[i]
                        cls_name = model.names[cls_id]
                        if cls_name == 'Elephant':
                            ele.setdefault(k, 0)
                            ele[k] += 1
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                            cv2.putText(frame, f'{cls_name}: {conf[i]:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        else:
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(frame, f'{cls_name}: {conf[i]:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                processed_frames.append(frame)
        t1 = time.time()
        rows = [np.concatenate(processed_frames[i:i+4], axis=1) for i in range(0, 16, 4)]
        multi = np.concatenate(rows, axis=0)
        cv2.namedWindow("BAGDOGRA-ELEPHANT ALARM TRIGGER", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("BAGDOGRA-ELEPHANT ALARM TRIGGER", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('BAGDOGRA-ELEPHANT ALARM TRIGGER', multi)
        # Check for exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release resources and destroy the window
    for cam in cameras:
        cam.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
