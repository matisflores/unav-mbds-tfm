import argparse
import shutil
import cv2
import queue
import os

from datetime import datetime

from mot.Detector import YOLODetector
from mot.Roi import Roi
from mot.MultiObjectTracker import MultiObjectTracker, track_from_motpy, previous_to_dict
from mot.Metrics import MetricDetections, MetricFPS, MetricTrackerErrors, MetricTrackers, MetricTrackersDelta, frame_cell_scores, frame_cell_traffic, stats_tracking_duration

from utils.Config import Config
from utils.DB import DB
from utils.EventListener import EventListener
from utils.Grid import Grid
from utils.Screen import Screen
from utils.Video import Video, VideoProcessor

def main(config_file):
    # Screen Layout
    SCREEN_PRIMARY = Screen('Video', width=0.75)
    SCREEN_STATS_1 = Screen('Stats 1', width=0.25, offset_x=0.75, height=0.27, resize=True, show=False)
    SCREEN_STATS_2 = Screen('Stats 2', width=0.25, offset_x=0.75, height=0.27, resize=True, offset_y=0.001, show=False)
    SCREEN_STATS_3 = Screen('Stats 3', width=0.25, offset_x=0.75, height=0.27, resize=True, offset_y=0.3, show=False)

    # Metrics
    METRIC_FPS = MetricFPS('FPS')
    METRIC_DETECTIONS = MetricDetections('Detections')
    METRIC_TRACKERS = MetricTrackers('Active Trackers')
    METRIC_ERRORS = MetricTrackerErrors('Tracker Errors')
    METRIC_TRACKERSDELTA = MetricTrackersDelta('Trackers Delta')

    # Events
    screen_events = queue.Queue()
    tracker_events = queue.Queue()
    track_events = queue.Queue()

    # Load configurations
    config = Config()
    config.load(config_file)

    # Output Directoy
    DIRECTORY_NAME = os.path.basename(config.get('source')) + '_' + datetime.now().strftime("%Y%m%d_%H%M%S")
    DIRECTORY_BASE = os.path.join(os.path.dirname(config.get('source')), DIRECTORY_NAME)
    os.makedirs(DIRECTORY_BASE, exist_ok=True)

    # Files
    FILE_DATABASE = DIRECTORY_BASE + '/tracking.db'
    FILE_FRAMES = DIRECTORY_BASE + '/frames'

    # Directories
    os.makedirs(FILE_FRAMES, exist_ok=True)

    # Backup files
    shutil.copy(config_file, DIRECTORY_BASE)
    shutil.copy(config.get('source'), DIRECTORY_BASE)    

    # Tracking options
    detection_rate: int = config.get('detection_rate', int)
    video_downscale: float = config.get('video_downscale', float)
    show_detections: bool = config.get('show_detections', bool)
    show_trackers: bool = config.get('show_trackers', bool)
    use_roi: bool = config.get('use_roi', bool)

    # Load video
    video = Video(config.get('source'))
    video.open()
    frame = video.read(downscale=video_downscale)

    # Divide frame
    grid = Grid(config.get('cell_size', int))
    grid.divide(frame)
    #frame = grid.plot(frame)

    # Roi selection
    if use_roi:
        roi = Roi(grid)
        roi.define(frame)
        #frame = roi.plot(frame)

    # Start tracking
    detector = YOLODetector()
    tracker = MultiObjectTracker.make(config.get('tracker'), video.fps)

    def on_track_event(event):
        db = DB(FILE_DATABASE)
        step = event['step']
        track = track_from_motpy(event['track']) if event['track'] is not None else None
        previous = track_from_motpy(event['previous']) if event['previous'] is not None else None
        diff = track.diff(previous) if track is not None and previous is not None else None
        timestamp = cv2.getTickCount()

        if track is None:
            return

        if not use_roi:
            cell = grid.in_cell(track.center)

            if cell is None:
                return

            _, _, _, id, _ = cell
            db.save_track(track._id, track.center, diff, id, step, None, track._score, timestamp)
            return
        
        current_cell = roi.in_cell(track.center)
        last_cell = roi.in_cell(previous.center) if previous is not None else None
        
        if current_cell is not None and last_cell is None:
            roi._count += 1
            _, _, _, id, _ = current_cell
            db.save_track(track._id, track.center, diff, id, step, roi._id, track._score, timestamp)
        elif current_cell is not None and last_cell is not None and current_cell != last_cell:
            _, _, _, id, _ = current_cell
            db.save_track(track._id, track.center, diff, id, step, roi._id, track._score, timestamp)
        elif current_cell is None and last_cell is not None:
            roi._count -= 1

    track_events_processor = EventListener(on_track_event, track_events)
    track_events_processor.start()

    def on_tracker_event(event):
        step = event['step']
        active_tracks = event['tracks']
        previous_tracks = previous_to_dict(event['previous'])

        for track in active_tracks:
            track_events.put({ 'step': step, 'track': track, 'previous': previous_tracks.get(track.id, None) })

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
        active_tracks, delta_trackers = tracker.step(detections=detections)
        tracker_events.put({ 'step': step, 'tracks': active_tracks, 'previous': tracker._previous_tracks })

        METRIC_TRACKERSDELTA.store(delta_trackers, step)

        error = tracker.error()
        if error is not None:
            METRIC_ERRORS.store(error[1], step)

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
                cv2.circle(frame, (int((track.box[0] + track.box[2])/2), int((track.box[1] + track.box[3])/2)), 2, (0,255,0), thickness=-1)

        # On End
        METRIC_FPS.stop(step)

        screen_events.put({'frame':frame,'step':step})

    def read_frame():
        return video.read(downscale=video_downscale, soft=True)

    video_processor = VideoProcessor(read_frame, on_frame)
    video_processor.start()

    # Process screen events on main thread
    while True:
        tmp = screen_events.get()

        if tmp is None:
            break

        frame = tmp['frame']
        step = tmp['step']

        if frame is None:
            break

        key = SCREEN_PRIMARY.show(frame, filename=FILE_FRAMES + f'/{step}.jpg')
        if key == ord('q'):
            break

    video_processor.stop()
    tracker_events_processor.stop()
    track_events_processor.stop()

    # Save metrics
    db = DB(FILE_DATABASE)
    METRIC_FPS.each(db.save_metrics)
    METRIC_DETECTIONS.each(db.save_metrics)
    METRIC_TRACKERS.each(db.save_metrics)
    METRIC_ERRORS.each(db.save_metrics)
    METRIC_TRACKERSDELTA.each(db.save_metrics)

    # Rewind video
    video.rewind()
    frame = video.read(downscale=video_downscale)

    # Save metrics image
    cv2.imwrite(DIRECTORY_BASE + '/cell_scores.jpg', frame_cell_scores(FILE_DATABASE, frame, grid, (50,0,0)))
    cv2.imwrite(DIRECTORY_BASE + '/cell_qty.jpg', frame_cell_traffic(FILE_DATABASE, frame, grid, (0,50,0)))
    cv2.imwrite(DIRECTORY_BASE + '/trackers_duration.jpg', stats_tracking_duration(FILE_DATABASE))
    cv2.imwrite(DIRECTORY_BASE + '/FPS.jpg', METRIC_FPS.plot())
    cv2.imwrite(DIRECTORY_BASE + '/detections.jpg', METRIC_DETECTIONS.plot())
    cv2.imwrite(DIRECTORY_BASE + '/trackers_errors.jpg', METRIC_ERRORS.plot())
    cv2.imwrite(DIRECTORY_BASE + '/trackers_active.jpg', METRIC_TRACKERS.plot())
    cv2.imwrite(DIRECTORY_BASE + '/trackers_delta.jpg', METRIC_TRACKERSDELTA.plot())

    # Close video
    video.release()
    
    # Show metrics
    SCREEN_STATS_1.show(METRIC_FPS.plot(), wait=False)
    #SCREEN_STATS_2.show(METRIC_DETECTIONS.plot(), wait=False)
    SCREEN_STATS_2.show(METRIC_ERRORS.plot(), wait=False)
    SCREEN_STATS_3.show(METRIC_TRACKERS.plot(), delay=0)

if __name__ == "__main__":
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='UNAV - Master en Big Data Science - Trabajo Final de Master')
    
    # Add command-line arguments
    parser.add_argument('--config', type=str, default='config.ini', help='Configuration File')
    
    # Parse command-line arguments
    args = parser.parse_args()

    # Call main function with command-line arguments
    main(args.config)
