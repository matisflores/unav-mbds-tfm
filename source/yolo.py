import os
import cv2
import fire
import sqlite3
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
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

def divide_frame(frame, zone_size):
    height, width, _ = frame.shape
    zones = []
    id = 0
    for y in range(0, height, zone_size):
        for x in range(0, width, zone_size):
            zone = frame[y:y+zone_size, x:x+zone_size]
            zones.append((x, y, zone, id, 0))
            id += 1
    print(f"Image: {height}x{width} - Zone Size: {zone_size} - Zones: {len(zones)}")
    return zones

def active_zone(frame, point, zones, zone_size):
    width = frame.shape[1]
    x, y = point[0],point[1]
    zone_index = ((y // zone_size) + 1) * (width // zone_size) + (x // zone_size)
    if zone_index < len(zones):
        print(f"Point: ({point}) - Zone: {zone_index} {zones[zone_index][3]}")
        return zone_index

    return None

def activate_zone(frame, point, zones, zone_size):
    width = frame.shape[1]
    x, y = point[0],point[1]
    zone_index = (y // zone_size) * (width // zone_size) + (x // zone_size)
    print(f"Point: ({x},{y}) - Zone: {zone_index} {zones[zone_index][3]}")
    if zone_index < len(zones):
        zones[zone_index] = (zones[zone_index][0], zones[zone_index][1], zones[zone_index][2], zones[zone_index][3], 1)

def plot_frame_with_zones(frame, zones, zone_size):
    zone_deactive = (255, 0, 0)
    zone_active = (255, 255, 255)
    frame_with_zones = frame.copy()
    for x, y, zone, id, active in zones:
        zone_color = zone_active if active == 1 else zone_deactive
        cv2.rectangle(frame_with_zones, (x, y), (x + zone_size, y + zone_size), zone_color, 1)
        cv2.putText(frame_with_zones, str(id), (x,y), cv2.FONT_HERSHEY_SIMPLEX, .3, zone_active, 1)
    return frame_with_zones

def create_table():
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS Points
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 tracking_id TEXT,
                 x INTEGER,
                 y INTEGER,
                 zone INTEGER,
                 datetime TEXT)''')
    conn.commit()
    conn.close()

def save_point(tracking_id, x, y, zone):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute("INSERT INTO Points (tracking_id, x, y, zone, datetime) VALUES (?, ?, ?, ?, ?)", (tracking_id, x, y, zone, now))
    conn.commit()
    conn.close()

def run(video_path: str= 'assets/1_original.mp4',
        #detect_labels,
        detection_rate: int = 1,
        video_downscale: float = 1.,
        confidence_threshold: float = 0.5,
        tracker_min_iou: float = 0.25,
        show_detections: bool = True,
        track_text_verbose: int = 0,
        viz_wait_ms: int = 1,
        zone_size = 40):
    
    # Create the table if it doesn't exist
    create_table()

    # setup detector, video reader and object tracker
    detector = YOLO('assets/yolov8n.pt')
    cap, cap_fps = read_video_file(video_path)
    tracker = MultiObjectTracker(
        dt=1 / cap_fps,
        tracker_kwargs={'max_staleness': 5},
        model_spec={'order_pos': 1, 'dim_pos': 2,
                    'order_size': 0, 'dim_size': 2,
                    'q_var_pos': 5000., 'r_var_pos': 0.1},
        matching_fn_kwargs={'min_iou': tracker_min_iou,
                            'multi_match_min_iou': 0.93})

    #zones
    ret, frame = cap.read()
    if not ret:
        exit(1)

    zones = divide_frame(frame, zone_size)

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
        active_tracks = tracker.active_tracks(min_steps_alive=detection_rate)

        # visualize and show detections and tracks
        if show_detections:
            for det in detections:
                draw_detection(frame, det)

        #active_zones = zones.copy()
        for track in active_tracks:
            center = (int((track.box[0]+track.box[2])/2),int((track.box[1]+track.box[3])/2))

            #activate_zone(frame, center, zones, zone_size)
            zone_id = active_zone(frame, center, zones, zone_size)
            if zone_id != None:
                x, y, zone, id, _ = zones[zone_id]
                save_point(track.id, center[0], center[1], id)

                #active_zones[zone_id] = x, y, zone, id, 1

                cv2.putText(frame, str(id), center, cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,0), 1)
            
            cv2.circle(frame, center, 1, (0,255,0), thickness=-1)
            draw_track(frame, track, thickness=2, text_at_bottom=True, text_verbose=track_text_verbose, random_color=False, fallback_color=(0,0,255))

        #plot zones
        #frame = plot_frame_with_zones(frame, active_zones, zone_size)
        frame = plot_frame_with_zones(frame, zones, zone_size)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Display Info
        drawText(frame, "FPS : {} ({}%)".format(str(int(fps)), int(fps/cap_fps*100)), (10, 30))

        cv2.imshow(f'Processed: {video_path}', frame)
        c = cv2.waitKey(viz_wait_ms)
        if c == ord('q'):
            break

        step += 1

    cap.release()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    fire.Fire(run)
