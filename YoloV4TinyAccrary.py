import cv2
import numpy as np
import time
import sys
import os

CONFIDENCE = 0.1
SCORE_THRESHOLD = 0.1
IOU_THRESHOLD = 0.1
scale_percent = 50

config_path = "yoloV4Tiny/yolov4-tiny-obj.cfg"
weights_path = "yoloV4Tiny/yolov4-tiny-obj_last.weights"
labels = open("yoloV4Tiny/obj.names").read().strip().split("\n")


#print(labels)
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")
def detech(image):

    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    net.setInput(blob)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    start = time.perf_counter()
    layer_outputs = net.forward(ln)
    time_took = time.perf_counter() - start
    sure = f"Gecen Sure: {time_took:.2f}s"

    font_scale = 1
    thickness = 1
    boxes, confidences, class_ids = [], [], []
    sayi = 0
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONFIDENCE:
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # print(detection.shape)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)

    red_tomato_list = []
    green_tomato_list = []
    red = 0
    green = 0
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            color = [int(c) for c in colors[class_ids[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
            text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
            # print(class_ids[i])
            if class_ids[i] == 1:
                red += 1
                red_tomato_list.append(f"{labels[class_ids[i]]}: {confidences[i]:.2f}")
            elif class_ids[i] == 0:
                green += 1
                green_tomato_list.append(f"{labels[class_ids[i]]}: {confidences[i]:.2f}")
            (text_width, text_height) = \
            cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
            text_offset_x = x
            text_offset_y = y - 5
            box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
            overlay = image.copy()
            cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
            image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=(0, 0, 0),
                        thickness=thickness)
            sayi += 1

    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    # cv2.imshow("yolov4-tiny", image)
    """
    print("KIRMIZI DOMATESLER VE DOĞRULUK ORANLARI")
    for i in red_tomato_list:
        print(i)

    print("Yesil DOMATESLER VE DOĞRULUK ORANLARI")
    for i in green_tomato_list:
        print(i)
    
    print("Bulunan domates sayisi : " + str(sayi))
    print("Bulunan Kırmızı Domates Sayısı : " + str(red))
    print("Bulunan Yesil Domates Sayısı : " + str(green))

    #print(sure)
    """

    return green,red

