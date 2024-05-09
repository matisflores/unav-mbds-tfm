import cv2
import fire
import sqlite3
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

def roi_stats():
    # Connect to the SQLite database
    conn = sqlite3.connect('data/tracking.db')
    c = conn.cursor()

    # Execute query to fetch distinct tracking IDs
    c.execute('''
        SELECT 
            tracker,
            MIN(timestamp) AS min_timestamp,
            MAX(timestamp) AS max_timestamp,
            (strftime('%s', MAX(timestamp)) - strftime('%s', MIN(timestamp))) AS timestamp_difference
        FROM 
            roi_tracks
        GROUP BY 
            tracker;
    ''')
    stats = c.fetchall()

    conn.close()

    return stats

def load_tracking_ids():
    # Connect to the SQLite database
    conn = sqlite3.connect('data/tracking.db')
    c = conn.cursor()

    # Execute query to fetch distinct tracking IDs
    c.execute("SELECT DISTINCT tracker FROM roi_tracks")
    tracking_ids = [row[0] for row in c.fetchall()]

    conn.close()

    return tracking_ids

def display_points_with_velocity(frame, tracking_ids):
    # Create a black image
    #width, height = 800, 600
    #black_screen = np.zeros((height, width, 3), dtype=np.uint8)

    # Connect to the SQLite database
    conn = sqlite3.connect('data/tracking.db')
    c = conn.cursor()

    for tracking_id in tracking_ids:
        # Generate a unique color for the tracking ID
        #color = np.random.randint(0, 255, size=(3,)).tolist()
        color = (0, 0, 255)

        # Retrieve points for the current tracking ID
        c.execute("SELECT * FROM roi_tracks WHERE tracker=? ORDER BY timestamp", (tracking_id,))
        points = c.fetchall()

        # Display points on the screen with velocity
        prev_x, prev_y = None, None
        for point in points:
            # Extract point information
            _, _, x, y, _, _ = point

            # Draw a circle at the point coordinates on the black screen
            cv2.circle(frame, (x, y), 10, color, -1)  # Use the unique color

            # Update previous point coordinates
            prev_x, prev_y = x, y

    conn.close()

    # Display the black screen with points
    cv2.imshow('Points', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run():
    # Show stats
    results = roi_stats()

    # Print the formatted results
    for row in results:
        tracker, min_timestamp, max_timestamp, timestamp_difference = row
        print(f"Tracker: {tracker}")
        print(f"Minimum Timestamp: {min_timestamp}")
        print(f"Maximum Timestamp: {max_timestamp}")
        print(f"Timestamp Difference (seconds): {timestamp_difference}")
        print()

    # Fetch the timestamp differences
    timestamp_differences = [row[3] for row in results]

    # Plot the distribution
    plt.hist(timestamp_differences, bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Timestamp Differences')
    plt.xlabel('Timestamp Difference (minutes)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    exit()

    # Load video
    video_path = 'assets/1_jail.mp4'
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

    frame_nro = 0
    frames_skip = 10
    while frame_nro < frames_skip:
        _, frame = cap.read()
        if not ret:
            print("Error: Unable to read video.")
            exit()

    # Load all tracking IDs from the SQLite table
    tracking_ids = load_tracking_ids()
    display_points_with_velocity(frame, tracking_ids)

if __name__ == '__main__':
    fire.Fire(run)
