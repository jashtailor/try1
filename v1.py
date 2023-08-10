import streamlit as st
import cv2
import numpy as np

def main():
    st.title("Live Object Detection with YOLO")

    # Load YOLO
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = f.read().splitlines()

    cap = cv2.VideoCapture(0)  # Open the default camera

    # Initialize OpenCV window
    window = st.image([])

    while True:
        ret, frame = cap.read()  # Read a frame from the webcam
        if not ret:
            break

        height, width, _ = frame.shape

        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_names = net.getUnconnectedOutLayersNames()
        outs = net.forward(layer_names)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in indexes:
            i = i[0]
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = np.random.uniform(0, 255, size=3)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

        # Convert OpenCV image to RGB format for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the resulting frame with object detection
        window.image(frame_rgb, channels="RGB")

    cap.release()

if __name__ == "__main__":
    main()
