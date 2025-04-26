from ultralytics import YOLO
import cv2 


# Load a model
model = YOLO("yolov8n-pose.pt")  # load an official model

activity_name = "SalsaSpin"

video_path = f"UCF101/train/{activity_name}/v_{activity_name}_g01_c03.avi"

cap = cv2.VideoCapture(video_path)


while True: 
    ret, frame = cap.read()    

    if ret: 
        results = model.track(frame)
        # Access the results
        for result in results:
            xy = result.keypoints.xy  # x and y coordinates
            xyn = result.keypoints.xyn  # normalized
            kpts = result.keypoints.data  # x, y, visibility (if available)


        cv2.imshow("output", results[0].plot())
    else: 
        print(ret, " Frame not found")


    if cv2.waitKey(1) & 0xFF == ord("q"): 
        break  


