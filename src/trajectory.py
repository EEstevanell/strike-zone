import time

import cv2
import numpy as np

from ransac import *
from utils import *


class ImageProcessor:
    @staticmethod
    def softened(frame, ksize=11, kernel=(3, 3)):
        return cv2.medianBlur(frame, ksize, kernel)

    @staticmethod
    def back_substraction(frame):
        background_substractor = cv2.createBackgroundSubtractorMOG2()
        return background_substractor.apply(frame)

    @staticmethod
    def get_median_frame(capture):
        frameIds = capture.get(cv2.CAP_PROP_FRAME_COUNT) * \
            np.random.uniform(size=30)

        # Store selected frames in an array
        frames = []
        for fid in frameIds:
            capture.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = capture.read()
            frames.append(frame)

        # Calculate the median along the time axis
        return np.median(frames, axis=0).astype(dtype=np.uint8)

    @staticmethod
    def video_back_substraction(frames, medianFrame):
        grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

        result_frames = []
        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Calculate absolute difference of current frame and
            # the median frame
            dframe = cv2.absdiff(frame, grayMedianFrame)
            # Treshold to binarize
            th, dframe = cv2.threshold(dframe, 50, 255, cv2.THRESH_BINARY)
            result_frames.append(dframe)

        # return frames without background
        return result_frames

    @staticmethod
    def apply(frame):
        frame = ImageProcessor.softened(frame)
        frame = ImageProcessor.back_substraction(frame)
        return frame


class BlobDetector:
    @staticmethod
    def frame_detector(frame):
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()

        params.minThreshold = 100
        params.maxThreshold = 300

        params.filterByColor = True
        params.blobColor = 255

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 150
        params.maxArea = 600

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.7

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.45

        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.01

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(frame)
        return keypoints

    @staticmethod
    def video_detector(cap):
        frames = []
        counter = 0
        while cap.isOpened():
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            keypoints = BlobDetector.frame_detector(frame)
            nframe = cv2.drawKeypoints(frame, keypoints, np.array(
                []), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow('frame', nframe)

            frames.append(([keypoint_to_coordinates(kp)
                          for kp in keypoints], counter))
            counter += 1

            if cv2.waitKey(1) == ord('q'):
                break

        return frames

    @staticmethod
    def video_detector_from_frames(frames):
        result = []
        counter = 0
        for frame in frames:
            keypoints = BlobDetector.frame_detector(frame)
            nframe = cv2.drawKeypoints(frame, keypoints, np.array(
                []), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # cv2.imshow('frame', nframe)

            if len(keypoints) > 0:
                result.append((keypoints, counter))

            counter += 1

            if cv2.waitKey(5) == ord('q'):
                break

        return result

    @staticmethod
    def unpack_video_blobs(blobs):
        items = [([keypoint_to_coordinates(kp) for kp in kpl], frame)
                 for kpl, frame in blobs]
        return [(*[list(t) for t in zip(*coordinates)], frame) for coordinates, frame in items]


class BlobFilter:
    pass


class Trajectory:
    def __init__(self, unpacked_video_blobs) -> None:
        self.x_distribution = [(x, frame)
                               for xs, _, frame in unpacked_video_blobs for x in xs]
        self.y_distribution = [(y, frame)
                               for _, ys, frame in unpacked_video_blobs for y in ys]

    def plot_x_distribution(self):
        import matplotlib.pyplot as plt

        y = [x[0] for x in self.x_distribution]
        x = [x[1] for x in self.x_distribution]

        plt.scatter(x, y)

        # Adding the title
        plt.title("X-Distribución")

        # Adding the labels
        plt.ylabel("coordenada x del blob")
        plt.xlabel("Número de cuadro")

        plt.show()

    def plot_y_distribution(self):
        import matplotlib.pyplot as plt

        y = [y[0] for y in self.y_distribution]
        x = [y[1] for y in self.y_distribution]

        plt.scatter(x, y)

        # Adding the title
        plt.title("Y-Distribución")

        # Adding the labels
        plt.ylabel("coordenada y del blob")
        plt.xlabel("Número de cuadro")

        plt.show()

    def detect_line(self, frame, x):
        line_ransac = RANSAC(
            n=2,
            t=1,
            d=2,
            k=500,
            model=Line(),
            loss=min_distance_loss,
            metric=count_out_of_geo_error
        )
        pass

    def detect_parab(self, frame, y):
        pass

    def detect_trajectory(self, frame, x, y):
        pass


class VideoTrajectory:
    pass


# result = BlobDetector.video_detector(r'D:\Work\UH-IA\Computer Vision\dataset\youtube\1.mp4')
cap = cv2.VideoCapture(r'D:\Work\UH-IA\Computer Vision\dataset\youtube\1.mp4')
medianFrame = ImageProcessor.get_median_frame(cap)

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

    # keypoints = BlobDetector.frame_detector(nframe)
    # nframe = cv2.drawKeypoints(nframe, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # nframe = ImageProcessor.back_substraction(frame)
    # cv2.imshow('frame', nframe)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()

fg_video = ImageProcessor.video_back_substraction(frames, medianFrame)
result = BlobDetector.video_detector_from_frames(fg_video)
video_blobs = BlobDetector.unpack_video_blobs(result)

trajectory = Trajectory(video_blobs)
trajectory.plot_x_distribution()
trajectory.plot_y_distribution()
