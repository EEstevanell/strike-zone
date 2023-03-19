import cv2
import torch

# Load the model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)

cap = cv2.VideoCapture(r'D:\Work\UH-IA\Computer Vision\dataset\videos\videos\video26.avi')

# Reset frame number to 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

frames = []
while cap.isOpened():
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        break
    
    # Load the image
    img = frame

    # Perform inference
    results = model(img)

    # Extract the bounding boxes for baseball balls (class 32)
    # class 34 = baseball bat
    # class 35 = baseball glove
    boxes = results.xyxy[0].numpy()
    baseball_boxes = boxes[boxes[:, 5] == 32]

    # Draw the bounding boxes on the image
    for box in baseball_boxes:
        x1, y1, x2, y2 = box[:4].astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)

    # Show the resulting image
    cv2.imshow('Result', img)
    
    frames.append(frame)
    if cv2.waitKey(1) == ord('q'):
        break
