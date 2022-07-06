import argparse
import cv2
import numpy as np
import time

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,help="image path giriniz.")
ap.add_argument("-w", "--weights", required=True,help="weights path giriniz.")
ap.add_argument("-c", "--config", required=True,help="config path giriniz.")
ap.add_argument("-n", "--names", required=True,help="names path giriniz.")
ap.add_argument("-cn", "--confidence", required=True,help="confidence giriniz.")
ap.add_argument("-st", "--score_th", required=True,help="score_th giriniz.")
ap.add_argument("-io", "--iou_th", required=True,help="iou_th giriniz.")
args = vars(ap.parse_args())


class Yolo:

    def __init__(self, conf, score, iou, config_path, weights_path, names_path, image_path):
        print("Starting detection image...")
        self.once_time = time.time()
        self.image_path = "images/"+image_path
        self.CONFIDENCE = conf
        self.SCORE_THRESHOLD = score
        self.IOU_THRESHOLD = iou
        self.config_path = "model/"+config_path
        self.weights_path = "model/"+weights_path
        self.labels = open("model/"+names_path).read().strip().split("\n")
        self.colors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype="uint8")
        self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)
        self.detection()

    def detection(self):
        image = cv2.imread(self.image_path)
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        ln = self.net.getLayerNames()
        ln = [ln[i - 1] for i in self.net.getUnconnectedOutLayers()]
        layer_outputs = self.net.forward(ln)
        boxes, confidences, class_ids = [], [], []
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.CONFIDENCE:
                    box = detection[:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.SCORE_THRESHOLD, self.IOU_THRESHOLD)
        if len(idxs) > 0:
            for i in idxs.flatten():
                text = f"{self.labels[class_ids[i]]}: {confidences[i]:.2f}"
                print(text)

    def __del__(self):
        print("ms : ",(time.time()-self.once_time)*1000)
        print("Finished detection image...")


if __name__ == "__main__":

    image_path = args["image"]
    weights_path = args["weights"]
    config_path = args["config"]
    names_path = args["names"]
    confidence = float(args["confidence"])
    score_th = float(args["score_th"])
    iou_th = float(args["iou_th"])
    yolo = Yolo(confidence, score_th, iou_th, config_path, weights_path, names_path, image_path)

