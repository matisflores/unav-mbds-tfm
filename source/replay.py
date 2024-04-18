import cv2
import fire
import sqlite3
from datetime import datetime
import numpy as np

def load_tracking_ids():
    # Connect to the SQLite database
    conn = sqlite3.connect('data.db')
    c = conn.cursor()

    # Execute query to fetch distinct tracking IDs
    c.execute("SELECT DISTINCT tracking_id FROM Points")
    tracking_ids = [row[0] for row in c.fetchall()]

    conn.close()

    return tracking_ids

def display_points_with_velocity(tracking_ids):
    # Create a black image
    width, height = 800, 600
    black_screen = np.zeros((height, width, 3), dtype=np.uint8)

    # Connect to the SQLite database
    conn = sqlite3.connect('data.db')
    c = conn.cursor()

    for tracking_id in tracking_ids:
        # Generate a unique color for the tracking ID
        color = np.random.randint(0, 255, size=(3,)).tolist()

        # Retrieve points for the current tracking ID
        c.execute("SELECT * FROM Points WHERE tracking_id=? ORDER BY datetime", (tracking_id,))
        points = c.fetchall()

        # Display points on the black screen with velocity
        prev_x, prev_y = None, None
        for point in points:
            # Extract point information
            _, _, x, y, _, _ = point

            # Draw a circle at the point coordinates on the black screen
            cv2.circle(black_screen, (x, y), 5, color, -1)  # Use the unique color

            # Update previous point coordinates
            prev_x, prev_y = x, y

    conn.close()

    # Display the black screen with points
    cv2.imshow('Points', black_screen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run():
    # Load all tracking IDs from the SQLite table
    tracking_ids = load_tracking_ids()
    display_points_with_velocity(tracking_ids)

if __name__ == '__main__':
    fire.Fire(run)
