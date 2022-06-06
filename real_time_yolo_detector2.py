import cv2
import numpy as np
import time

w_p="/home/mef/Documents/dogu/YOLO-Real-Time-Object-Detection/yolov3-tiny.weights"
c_p="/home/mef/Documents/dogu/YOLO-Real-Time-Object-Detection/yolov3-tiny.cfg"
# Load Yolo
net = cv2.dnn.readNet(w_p, c_p)
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
cap = cv2.VideoCapture("video.mp4")

arac = 0 
cars_counter_list = list()
font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
offset=10       
line_height=650 



while True:
    _, frame = cap.read()
    frame_id += 1
    cv2.line(frame, (0, 650), (2000, 650), (0,255,0), 4)

    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[3] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 1.8)
                y = int(center_y - h / 1.8)

                boxes.append([x, y, w, h])

                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            cars_counter_list.append(y)
                  
            #counter
            if y<(line_height+offset) and y>(line_height-offset):
                arac=arac+1
                cars_counter_list.remove(y)

            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 2, color, 2)

    cv2.putText(frame, "Toplam gecen araba: " + str(arac), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 170, 0), 2)

    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 2, (0, 0, 0), 3)
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()