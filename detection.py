import cv2
import YoloV4Tiny as yolo

cap = cv2.VideoCapture("video/1.mp4")

cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

while True:

    ret, frame = cap.read()

    c = yolo.detech(frame)
    cv2.imshow("a", c)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()