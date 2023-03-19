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
        # cv2.imshow('frame2', grayMedianFrame)

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


class Trajectory:
    def __init__(self, unpacked_video_blobs) -> None:
        self.data = unpacked_video_blobs
        self.x_distribution = [(x, frame)
                               for xs, _, frame in unpacked_video_blobs for x in xs]
        self.y_distribution = [(y, frame)
                               for _, ys, frame in unpacked_video_blobs for y in ys]
        self.x_regressor = None
        self.y_regressor = None
        self.detected_points = None

    def plot_x_distribution(self, predictor = None ,pids = None):
        import matplotlib.pyplot as plt

        source = self.x_distribution if pids is None else np.array(self.x_distribution)[pids]
        y = [x[0] for x in source]
        X = [x[1] for x in source]

        # Adding the title
        plt.title("X-Distribución")

        # Adding the labels
        plt.ylabel("coordenada x del blob")
        plt.xlabel("Número de cuadro")

        if predictor is not None:
            plt.style.use("seaborn-darkgrid")

            min_X = min(X)
            max_X = max(X)

            x_axis = np.linspace(min_X, max_X, int(max_X - min_X))
            plt.plot(x_axis, predictor.predict(x_axis), c="peru")
        
        plt.scatter(X, y)
        plt.show()

    def plot_y_distribution(self, predictor = None, pids = None):
        import matplotlib.pyplot as plt

        source = self.y_distribution if pids is None else np.array(self.y_distribution)[pids]
        y = [y[0] for y in source]
        X = [y[1] for y in source]

        # Adding the title
        plt.title("Y-Distribución")

        # Adding the labels
        plt.ylabel("coordenada y del blob")
        plt.xlabel("Número de cuadro")

        if predictor is not None:
            plt.style.use("seaborn-darkgrid")

            min_X = min(X)
            max_X = max(X)

            x_axis = np.linspace(min_X, max_X, int(max_X - min_X))
            plt.plot(x_axis, predictor.predict(x_axis), c="peru")
            
        plt.scatter(X, y)
        plt.show()

    def _detect_line(self, plot=False):
        regressor = RANSAC(
            n=2,
            t=1,
            d=2,
            k=500,
            model=Line(),
            loss=min_distance_loss,
            metric=count_out_of_geo_error
        )

        y = [x[0] for x in self.x_distribution] # frames
        X = [x[1] for x in self.x_distribution] # x position from x distribution

        point_to_pos = dict()
        for i in range(len(y)):
            point_to_pos[(X[i], y[i])] = i

        regressor.fit(X, y, sequential=True)

        if plot:
            import matplotlib.pyplot as plt
            
            plt.title(f"{regressor.best_fit} Matches: {len(X) - regressor.best_error}")

            min_X = min(X)
            max_X = max(X)

            line = np.linspace(min_X, max_X, int(max_X - min_X))
            plt.plot(line, regressor.predict(line), c="peru")
            plt.scatter(X, y)

            plt.show()

        return regressor, [point_to_pos[(regressor.inlier_X[i], regressor.inlier_y[i])] for i in range(len(regressor.inlier_X))] if regressor.inlier_X is not None else None

    def _detect_parab(self, pids, plot=False):
        regressor = RANSAC(
            n=3, # amount of points to train the parabola
            t=1, # distance threshold
            d=3, # minimum amount of matching points to the parabola
            k=500, # max amount of iterations (retries) if not sequential training
            acc_error=2,
            model=Parabola(), 
            loss=min_distance_loss, 
            metric=count_out_of_geo_error
        )

        # only get fitted points by previous line
        source = np.array(self.y_distribution)
        y = [x[0] for x in source[pids]] # frames
        X = [x[1] for x in source[pids]] # y position from y distribution
    

        point_to_pos = dict()
        for i in range(len(y)):
            point_to_pos[(X[i], y[i])] = i

        regressor.fit(X, y, sequential=True)

        if plot:
            import matplotlib.pyplot as plt

            plt.title(f"{regressor.best_fit} Matches: {len(X) - regressor.best_error}")
            plt.style.use("seaborn-darkgrid")

            min_X = min(X)
            max_X = max(X)

            line = np.linspace(min_X, max_X, int(max_X - min_X))
            plt.plot(line, regressor.predict(line), c="peru")
            plt.scatter(X, y)
        
            plt.show()
        return regressor, [point_to_pos[(regressor.inlier_X[i], regressor.inlier_y[i])] for i in range(len(regressor.inlier_X))] if regressor.inlier_X is not None else None
    
    def detect_trajectory(self, tries=10):
        for _ in range(tries):
            line, pids = self._detect_line()
            parab, p_pids = self._detect_parab(pids)
            if (parab is not None):
                self.detected_points = np.array(pids)[p_pids]
                self.x_regressor, self.y_regressor = (line, parab)
        return self

    def get_first_ocurrence_frame(self):
        if (self.x_distribution is None):
            raise Exception("No data provided")
            
        if (self.detected_points is None):
            raise Exception("No trajectory detected")
        
        return self.x_distribution[self.detected_points[0]][1]

    def predict(self, frame) -> Tuple[int, int]:
        px = self.x_regressor.predict([frame])[0]
        py = self.y_regressor.predict([frame])[0]
        return px, py

class VideoTrajectory:
    def __init__(self, trajectory=None) -> None:
        self.trajectory = trajectory

    def compute_trajectory(self, video_path):
        cap, frames = load_video(video_path)

        self.capture = cap
        self.frames = frames

        # Obtain frame size information using get() method
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_size = (self.frame_width, self.frame_height)
        self.fps = 20

        medianFrame = ImageProcessor.get_median_frame(cap)
        cap.release()

        fg_video = ImageProcessor.video_back_substraction(frames, medianFrame)
        result = BlobDetector.video_detector_from_frames(fg_video)
        video_blobs = BlobDetector.unpack_video_blobs(result)

        trajectory = Trajectory(video_blobs)
        self.trajectory = trajectory.detect_trajectory(tries=2)

        # # write the output
        # output = cv2.VideoWriter(r'D:\Work\UH-IA\Computer Vision\o.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 20, frame_size)
        # for frame in frames:
        #     output.write(frame)
        # output.release()

    def compute_trajectory_image(self, initial_frame=None):
        fframe = initial_frame if initial_frame is not None else self.trajectory.get_first_ocurrence_frame()
        image = self.frames[fframe]

        for i in range(fframe, len(self.frames)):
            # Draw a point at (x, y) position
            x,y = self.trajectory.predict(i)
            cv2.circle(image, (int(x), int(y)), 15, (0, 0, 255), 2)

        return image
    
    def export_trajectory_image(self, filename, initial_frame=None):
        image = self.compute_trajectory_image(initial_frame)
        cv2.imwrite(filename, image)
    
    def show_trajectory_image(self, initial_frame=None):
        image = self.compute_trajectory_image(initial_frame)

        # Display the frame
        cv2.imshow('frame', image)
        cv2.waitKey()
    
    def compute_trajectory_video(self, filename=None, initial_frame=None):
        fframe = initial_frame if initial_frame is not None else self.trajectory.get_first_ocurrence_frame()
        output = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc('M','J','P','G'), 20, self.frame_size) if filename is not None else None
        new_frames = []
        data_points = []
        for fi in range(len(self.frames)):
            frame = self.frames[fi]
            if fi >= fframe:
                x,y = self.trajectory.predict(fi)
                data_points.append((x,y))

            for dp in data_points:
                x,y = dp
                cv2.circle(frame, (int(x), int(y)), 10, (0, 0, 255), 2)

            new_frames.append(frame)
            if output is not None:
                output.write(frame)
            
        return new_frames

if __name__ == "__main__":
    v_trajectory = VideoTrajectory()
    v_trajectory.compute_trajectory(r'D:\Work\UH-IA\Computer Vision\dataset\youtube\1.mp4')
    # v_trajectory.export_trajectory_image(r'D:\Work\UH-IA\Computer Vision\output\img26.png')
    v_trajectory.compute_trajectory_video(r'D:\Work\UH-IA\Computer Vision\output\1.avi')
    