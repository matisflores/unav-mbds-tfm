import argparse
import cv2
import queue
import os

from datetime import datetime

from mot.Algorithm import Algorithm
from mot.Detector import YOLODetector
from mot.Roi import Roi
from mot.MOTKalmanTracker import MOTKalmanTracker
from mot.Metrics import MetricDetections, MetricFPS, MetricTrackers

from utils.Config import Config
from utils.DB import DB
from utils.EventListener import EventListener
from utils.Grid import Grid
from utils.Screen import Screen
from utils.Video import Video

def main(config_file):
    # Screen Layout
    SCREEN_PRIMARY = Screen('Video', width=0.75)
    SCREEN_STATS_1 = Screen('Stats 1', width=0.25, offset_x=0.75, height=0.27, resize=True)
    SCREEN_STATS_2 = Screen('Stats 2', width=0.25, offset_x=0.75, height=0.27, resize=True, offset_y=0.001)
    SCREEN_STATS_3 = Screen('Stats 3', width=0.25, offset_x=0.75, height=0.27, resize=True, offset_y=0.3)

    # Metrics
    METRIC_FPS = MetricFPS('FPS')
    METRIC_DETECTIONS = MetricDetections('Detections')
    METRIC_TRACKERS = MetricTrackers('Active Trackers')

    # Events
    screen_events = queue.Queue()
    tracker_events = queue.Queue()
    tracking_events = queue.Queue()

    # Load configurations
    config = Config()
    config.load(config_file)

    # Database
    db_file = config.data_dir + '/' + os.path.basename(config.source) + '_' + datetime.now().strftime("%H%M%S") + '_tracking.db'

    # Tracking options
    detection_rate: int = 1
    video_downscale: float = 1
    show_detections: bool = True
    show_trackers: bool = True
    use_roi: bool = False

    # Load video
    video = Video(config.source)
    video.open()
    frame = video.read(downscale=video_downscale)

    # Divide frame
    cell_size = int(config.cell_size)
    grid = Grid(cell_size)
    grid.divide(frame)
    #frame = grid.plot(frame)

    # Roi selection
    if use_roi:
        roi = Roi(grid)
        roi.define(frame)
        #frame = roi.plot(frame)

    # Start tracking
    detector = YOLODetector()
    tracker = MOTKalmanTracker(video.fps)

    track_by_roi = set()

    def on_tracking_event(event):
        track = event['track']
        step = event['step']

        if use_roi:
            db = DB(db_file)

            cell = roi.in_zone(track.center)
            if cell is not None:
                if track._id not in track_by_roi:
                    track_by_roi.add(track._id)
                    roi._count = len(track_by_roi)
                
                x, y, frame_part, id, _ = cell
                db.save_tracker_rois(track._id, '1', id, step)
            else:
                if track._id in track_by_roi:
                    track_by_roi.remove(track._id)
                    roi._count = len(track_by_roi)

    tracking_events_processor = EventListener(on_tracking_event, tracking_events)
    tracking_events_processor.start()

    def on_tracker_event(event):
        active_tracks = event['tracks']
        step = event['step']

        for track in active_tracks:
            tracking_events.put({ 'step': step, 'track': track })

        METRIC_TRACKERS.store(len(active_tracks), step)

    tracker_events_processor = EventListener(on_tracker_event, tracker_events)
    tracker_events_processor.start()

    # Tracking
    def on_frame(frame, step: int):
        # On Start
        METRIC_FPS.start()

        # Detect objects
        detections = []
        if step % detection_rate == 0:
            detections = detector.detect(frame)
            METRIC_DETECTIONS.store(len(detections), step)

        # Track detected objects
        active_tracks = tracker.step(detections=detections)
        tracker_events.put({ 'step': step, 'tracks': active_tracks })

        # Show roi
        if use_roi:
            frame = roi.plot(frame)

        # Show detections
        if show_detections:
            for det in detections:
                cv2.rectangle(frame, (int(det.box[0]), int(det.box[1])), (int(det.box[2]), int(det.box[3])), (255, 0, 0), 1)

        # Show trackers
        if show_trackers:
            for track in active_tracks:
                cv2.circle(frame, track.center, 2, (0,255,0), thickness=-1)

        # On End
        METRIC_FPS.stop(step)

        screen_events.put(frame)

    def read_frame():
        return video.read(downscale=video_downscale, soft=True)

    algorithm = Algorithm(read_frame, on_frame)
    algorithm.start()

    # Process screen events on main thread
    step = 0
    while True:
        frame = screen_events.get()

        if frame is None:
            break

        key = SCREEN_PRIMARY.show(frame)
        if key == ord('q'):
            break

        step += 1

    # Save metrics
    db = DB(db_file)
    def save_metric(metric, step, value):
        db.save_metrics(metric, step, value)

    METRIC_FPS.each(save_metric)
    METRIC_DETECTIONS.each(save_metric)
    METRIC_TRACKERS.each(save_metric)

    # Show metrics
    SCREEN_STATS_1.show(METRIC_FPS.plot(), wait=False)
    SCREEN_STATS_2.show(METRIC_DETECTIONS.plot(), wait=False)
    SCREEN_STATS_3.show(METRIC_TRACKERS.plot(), delay=0)

    algorithm.stop()
    video.release()
    tracker_events_processor.stop()
    tracking_events_processor.stop()

if __name__ == "__main__":
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='UNAV - Master en Big Data Science - Trabajo Final de Master')
    
    # Add command-line arguments
    parser.add_argument('--config', type=str, default='config.ini', help='Configuration File')
    
    # Parse command-line arguments
    args = parser.parse_args()

    # Call main function with command-line arguments
    main(args.config)
