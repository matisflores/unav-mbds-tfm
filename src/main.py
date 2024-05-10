
import argparse
import cv2

from config.config import Config
from data.db import DB
from mot.Detector import YOLODetector
from mot.Grid import Grid
from mot.Roi import Roi
from mot.Tracker import MOTKalmanTracker

def main(config_file):
    # Load configurations
    config = Config()
    config.load(config_file)

    # Tracking options
    detection_rate: int = 1
    video_downscale: float = 1.
    show_detections: bool = True

    # Open DB
    db = DB()

    # Load video
    video_path = config.source
    video_cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not video_cap.isOpened():
        print("Error: Unable to open video.")
        exit()

    # Get video FPS
    video_fps = float(video_cap.get(cv2.CAP_PROP_FPS))

    # Read first frame
    ret, frame = video_cap.read()
    if not ret:
        print("Error: Unable to read video.")
        exit()

    # Divide frame
    grid = Grid()
    grid.divide(frame)
    frame = grid.plot(frame)

    # Roi selection
    roi = Roi(grid)
    roi.define(frame)
    frame = roi.plot(frame)

    # Start tracking
    detector = YOLODetector()
    tracker = MOTKalmanTracker(video_fps)

    def on_step(active_tracks):
        for track in active_tracks:
            active = roi.in_zone(track.center)
            color = (0,0,255) if active else (0,255,0)
            cv2.circle(frame, track.center, 2, color, thickness=-1)

            if active:
                db.save_tracker_rois(track._id, roi._roi_cells)

    step = 0
    while True:
        ret, frame = video_cap.read()
        if not ret:
            break

        # Start timer
        timer = cv2.getTickCount()

        # Downscale frame
        #frame = cv2.resize(frame, fx=video_downscale, fy=video_downscale, dsize=None, interpolation=cv2.INTER_AREA)

        # Detect objects
        detections = [] if step % detection_rate != 0 else detector.detect(frame)

        # Track detected objects
        active_tracks = tracker.step(detections=detections)

        on_step(active_tracks)

        # Show roi
        frame = roi.plot(frame)

        # visualize detections
        if show_detections:
            for det in detections:
                cv2.rectangle(frame, (int(det.box[0]), int(det.box[1])), (int(det.box[2]), int(det.box[3])), (255, 0, 0), 1)


        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.putText(frame, "FPS : {} ({}%)".format(str(int(fps)), int(fps/video_fps*100)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 170, 50), 3)

        # Show frame
        cv2.imshow(f'Processing: {video_path}', frame)
        c = cv2.waitKey(1)
        if c == ord('q'):
            break

        step += 1

    video_cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='Description of your program')
    
    # Add command-line arguments
    parser.add_argument('--config', type=str, default='config/config.ini', help='Configuration File')
    
    # Parse command-line arguments
    args = parser.parse_args()

    # Call main function with command-line arguments
    main(args.config)
