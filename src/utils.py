import cv2

def keypoint_to_coordinates(kp):
    return (kp.pt[0], kp.pt[1])

def load_video(path):
    cap = cv2.VideoCapture(path)

    # Reset frame number to 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frames.append(frame)
        if cv2.waitKey(1) == ord('q'):
            break

    return cap, frames