
import argparse
import cv2
import queue

from config.config import Config
from data.db import DB
from mot.Detector import YOLODetector
from mot.Grid import Grid
from mot.Roi import Roi
from mot.Tracker import MOTKalmanTracker
from mot.Track import Track
from utils.Worker import Worker
from utils.Layout import Layout, Screen

def main(config_file):
    # Load configurations
    config = Config()
    config.load(config_file)

    # Tracking options
    detection_rate: int = 1
    video_downscale: float = 1.
    show_detections: bool = True

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

    # Screen Layout
    SCREEN_PRIMARY = 'primary'
    SCREEN_STATS_1 = 'stats_1'
    SCREEN_STATS_2 = 'stats_2'
    SCREEN_STATS_3 = 'stats_3'
    screens = {
        SCREEN_PRIMARY: Screen(f'Processing: {video_path}', width=0.75, height=1),
        SCREEN_STATS_1: Screen('Stadistica 1', width=0.25, offset_x=0.75, height=0.27, resize=True),
        SCREEN_STATS_2: Screen('Stadistica 2', width=0.25, offset_x=0.75, height=0.27, resize=True, offset_y=0.001),
        SCREEN_STATS_3: Screen('Stadistica 3', width=0.25, offset_x=0.75, height=0.27, resize=True, offset_y=0.3),
    }
    screen = Layout(screens)

    # Divide frame
    grid = Grid()
    grid.divide(frame)
    #frame = grid.plot(frame)

    # Roi selection
    roi = Roi(grid)
    roi.define(frame)
    frame = roi.plot(frame)

    # Start tracking
    detector = YOLODetector()
    tracker = MOTKalmanTracker(video_fps)
    events = queue.Queue()

    def on_active(events: queue.Queue):
        db = DB()

        while True:
            ev = events.get()

            if ev is None:
                break

            track = ev.get('track', None)
            roi = ev.get('roi', None)

            if track is None or roi is None:
                continue

            db.save_tracker_rois(track._id, roi._roi_cells)

    analyzer = Worker(on_active, args=(events,))
    analyzer.start()

    def on_step(detections: list[any], active_tracks: list[Track]):
        for track in active_tracks:
            active = roi.in_zone(track.center)
            color = (0,0,255) if active else (0,255,0)
            cv2.circle(frame, track.center, 2, color, thickness=-1)

            if active:
                events.put({
                    'track': track,
                    'roi': roi
                })

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

        on_step(detections, active_tracks)

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
        c = screen.show(SCREEN_PRIMARY, frame, 1)
        if c == ord('q'):
            break

        step += 1

    video_cap.release()

    events.put(None)
    analyzer.join()
    screen.close()



if __name__ == "__main__":
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='Description of your program')
    
    # Add command-line arguments
    parser.add_argument('--config', type=str, default='config/config.ini', help='Configuration File')
    
    # Parse command-line arguments
    args = parser.parse_args()

    # Call main function with command-line arguments
    main(args.config)
