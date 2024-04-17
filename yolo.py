import os
import cv2
import fire
from ultralytics import YOLO

from motpy.core import Detection
from motpy.testing_viz import draw_detection, draw_track
from motpy.tracker import MultiObjectTracker

def read_video_file(video_path: str):
    video_path = os.path.expanduser(video_path)
    cap = cv2.VideoCapture(video_path)
    video_fps = float(cap.get(cv2.CAP_PROP_FPS))
    return cap, video_fps

def drawText(frame, txt, location, color=(50, 170, 50)):
    cv2.putText(frame, txt, location, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

def run(video_path: str,
        #detect_labels,
        detection_rate: int = 1,
        video_downscale: float = 1.,
        confidence_threshold: float = 0.5,
        tracker_min_iou: float = 0.25,
        show_detections: bool = True,
        track_text_verbose: int = 0,
        viz_wait_ms: int = 1):
    # setup detector, video reader and object tracker
    detector = YOLO('yolov8n.pt')
    cap, cap_fps = read_video_file(video_path)
    tracker = MultiObjectTracker(
        dt=1 / cap_fps,
        tracker_kwargs={'max_staleness': 5},
        model_spec={'order_pos': 1, 'dim_pos': 2,
                    'order_size': 0, 'dim_size': 2,
                    'q_var_pos': 5000., 'r_var_pos': 0.1},
        matching_fn_kwargs={'min_iou': tracker_min_iou,
                            'multi_match_min_iou': 0.93})

    step = 0
    while True:
        ret, frame = cap.read()
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

        # track detected objects
        _ = tracker.step(detections=detections)
        active_tracks = tracker.active_tracks(min_steps_alive=3)

        # visualize and show detections and tracks
        if show_detections:
            for det in detections:
                draw_detection(frame, det)

        for track in active_tracks:
            draw_track(frame, track, thickness=1, text_at_bottom=True, text_verbose=track_text_verbose, random_color=False, fallback_color=(0,0,255))

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Display Info
        drawText(frame, "FPS : {} ({}%)".format(str(int(fps)), int(fps/cap_fps*100)), (10, 30))

        cv2.imshow('frame', frame)
        c = cv2.waitKey(viz_wait_ms)
        if c == ord('q'):
            break

        step += 1

if __name__ == '__main__':
    fire.Fire(run)
