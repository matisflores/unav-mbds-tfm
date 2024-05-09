
import argparse
import cv2
from ultralytics import YOLO

from config.config import Config
from data.db import DB
from mot.Zones import Zones
from mot.Roi import Roi
from motpy.core import Detection
from motpy.tracker import MultiObjectTracker

def main(config_file):
    # Load configurations
    config = Config()
    config.load(config_file)

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
    zones = Zones()
    zones.divide(frame)
    #frame = zones.plot(frame)

    # Roi selection
    roi = Roi(zones)
    roi.define(frame)
    frame = roi.plot(frame)

    # Start tracking
    video_downscale: float = 1.
    show_detections: bool = True
    tracker_min_iou: float = 0.25
    detection_rate: int = 1
    confidence_threshold: float = 0.5

    detector = YOLO(config.path_assets_dir + '/yolov8n.pt')
    tracker = MultiObjectTracker(
        dt=1 / video_fps,
        tracker_kwargs={'max_staleness': 5},
        model_spec={'order_pos': 1, 'dim_pos': 2, 'order_size': 0, 'dim_size': 2, 'q_var_pos': 5000., 'r_var_pos': 0.1},
        matching_fn_kwargs={'min_iou': tracker_min_iou, 'multi_match_min_iou': 0.93})

    step = 0
    while True:
        ret, frame = video_cap.read()
        if not ret:
            break

        # Start timer
        timer = cv2.getTickCount()

        #frame = cv2.resize(frame, fx=video_downscale, fy=video_downscale, dsize=None, interpolation=cv2.INTER_AREA)

        if step == 0 or step % detection_rate == 0:
            # detect objects in the frame
            results = detector(frame, verbose=False)

            # Parse results
            detections = [Detection(box=b, score=s, class_id=l) for b, s, l in zip(results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.conf.cpu().numpy(), results[0].boxes.cls.cpu().numpy().astype(int))]
            detections = [ i for i in detections if i.class_id == 0 and i.score >= confidence_threshold]
        else:
            detections = []

        # visualize and show detections and tracks
        if show_detections:
            for det in detections:
                cv2.rectangle(frame, (int(det.box[0]), int(det.box[1])), (int(det.box[2]), int(det.box[3])), (255, 0, 0), 1)

        # track detected objects
        _ = tracker.step(detections=detections)
        active_tracks = tracker.active_tracks(min_steps_alive=3)

        for track in active_tracks:
            center = (int((track.box[0]+track.box[2])/2),int((track.box[1]+track.box[3])/2))

            active = roi.in_zone(center)
            color = (0,0,255) if active else (0,255,0)
            cv2.circle(frame, center, 2, color, thickness=-1)

            if active:
                #when person is in ROI
                db.save_tracker_rois(track.id, roi._roi_zones)

        # Show roi
        frame = roi.plot(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Display Info
        cv2.putText(frame, "FPS : {} ({}%)".format(str(int(fps)), int(fps/video_fps*100)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 170, 50), 3)

        cv2.imshow(f'Processed: {video_path}', frame)
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
