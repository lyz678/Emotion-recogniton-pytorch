import cv2
import numpy as np
import time


detection_model_path = 'haarcascade_frontalface_default.xml'
face_detection = cv2.CascadeClassifier(detection_model_path)
file_model = 'miniXception.sim.onnx'
net = cv2.dnn.readNet(file_model)

EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised","neutral"]

# starting video streaming
cv2.namedWindow('your_face')
camera = cv2.VideoCapture(0)

while True:
    start_time = time.time()

    frame = camera.read()[1]
    frame = cv2.resize(frame, (300, int(frame.shape[0] * (300 / frame.shape[1]))))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    frameClone = frame.copy()
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0

        roi = (roi - 0.5) * 2
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=0)
        
        net.setInput(roi)
        preds = net.forward()[0]
        prob = np.max(preds)
        label = EMOTIONS[preds.argmax()]
        cv2.putText(frameClone, label + " {:.2f}%".format(prob * 100), (fX, fY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),(0, 0, 255), 2)
    else:
        cv2.putText(frameClone, "no face", (50,120),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
 

     # Calculate FPS
    elapsed_time = time.time() - start_time
    fps = 1.0 / elapsed_time
    cv2.putText(frameClone, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow('your_face', frameClone)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
