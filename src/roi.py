import cv2
import sqlite3
import uuid

# Function to divide frame into zones
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

# Mouse callback function for ROI selection
def select_roi(event, x, y, flags, param):
    global roi_selected, roi_start, roi_end, frame, zones, roi_list
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_selected = True
        roi_start = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and roi_selected:
        roi_end = (x, y)
        frame_copy = frame.copy()  # Make a copy to avoid drawing on original frame
        cv2.rectangle(frame_copy, roi_start, (x, y), (0, 255, 0), 2)
        cv2.imshow('Select ROI', frame_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        roi_selected = False
        cv2.rectangle(frame, roi_start, (x, y), (0, 255, 0), 2)
        cv2.imshow('Select ROI', frame)
        
        # Extract selected zones and draw rectangles for each selected zone
        if roi_start is not None and roi_end is not None:
            x1, y1 = roi_start
            x2, y2 = roi_end

            selected_zones = [zone for zone in zones if ((x1 <= zone[0] and zone[0] < x2) or (x1 <= zone[0] + zone_size and zone[0] + zone_size < x2)) and
                                                          ((y1 <= zone[1] and zone[1] < y2) or (y1 <= zone[1] + zone_size and zone[1] + zone_size < y2))]
            for zone in selected_zones:
                x, y, _, _, _ = zone
                cv2.rectangle(frame, (x, y), (x+zone_size, y+zone_size), (0, 0, 255), 2)
            
            # Save ROI and selected zones to database
            save_roi_to_database(selected_zones)

            # Display the frame with selected zones
            cv2.imshow('Selected Zones', frame)
            roi_list.append((uuid.uuid4(), selected_zones))  # Add ROI and selected zones to the list of ROIs

# Function to save ROI and selected zones to the database
def save_roi_to_database(selected_zones):
    conn = sqlite3.connect('rois.db')
    c = conn.cursor()

    # Create table if not exists
    c.execute('''CREATE TABLE IF NOT EXISTS rois
                 (id TEXT PRIMARY KEY, selected_zones TEXT)''')

    # Generate unique ID for the ROI
    roi_id = str(uuid.uuid4())

    # Insert ROI and selected zones into the database
    c.execute("INSERT INTO rois (id, selected_zones) VALUES (?, ?)", (roi_id, str(selected_zones)))

    conn.commit()
    conn.close()

# Load video
video_path = 'assets/1_original.mp4'
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Unable to open video.")
    exit()

# Read first frame
ret, frame = cap.read()
if not ret:
    print("Error: Unable to read video.")
    exit()

# Initialize variables for ROI selection
roi_selected = False
roi_start = None
roi_end = None
zone_size = 40 # Change this value as needed
zones = divide_frame(frame, zone_size)
roi_list = []

# Create window and set mouse callback
cv2.namedWindow('Select ROI')
cv2.setMouseCallback('Select ROI', select_roi)

# Display first frame with zones
for zone in zones:
    x, y, _, _, _ = zone
    cv2.rectangle(frame, (x, y), (x+zone_size, y+zone_size), (255, 0, 0), 1)

cv2.imshow('Select ROI', frame)
cv2.waitKey(0)

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
