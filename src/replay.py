import argparse
import cv2
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from utils.Grid import Grid
from utils.Video import Video
from utils.Config import Config
from utils.DB import DB

'''
def roi_stats():
    # Connect to the SQLite database
    conn = sqlite3.connect('data/tracking.db')
    c = conn.cursor()

    # Execute query to fetch distinct tracking IDs
    c.execute('' '
        SELECT 
            tracker,
            MIN(timestamp) AS min_timestamp,
            MAX(timestamp) AS max_timestamp,
            (strftime('%s', MAX(timestamp)) - strftime('%s', MIN(timestamp))) AS timestamp_difference
        FROM 
            roi_tracks
        GROUP BY 
            tracker;
    '' ')
    stats = c.fetchall()

    conn.close()

    return stats
'''

def main(config_file):
    '''
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
    '''

    # Load configurations
    config = Config()
    config.load(config_file)

    # Database
    db = DB(config.data_dir + '/tracking.db')

    # Load video
    video = Video(config.source)
    video.open()
    frame = video.read(skip=100)

    # Divide frame
    cell_size = int(config.cell_size)
    grid = Grid(cell_size)
    grid.divide(frame)
    #frame = grid.plot(frame)

    # Load all tracking IDs from the SQLite table
    tracking_ids = db.load_tracking_ids()

    # Create a black image
    #height, width, _ = frame.shape
    #frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    for tracking_id in tracking_ids:
        # Generate a unique color for the tracking ID
        color = np.random.randint(0, 255, size=(3,)).tolist()
        #color = (0, 0, 255)

        # Retrieve points for the current tracking ID
        points = db.load_tracking_points(tracking_id)

        # Display points on the screen with velocity
        for point in points:
            # Extract point information
            _,_,id,_ = point

            cell = grid.cell(int(id))

            if cell is None:
                continue

            x,y,_,_,_ = cell
            x = int(x)
            y = int(y)
            x += int(config.cell_size)/2
            y += int(config.cell_size)/2
            # Draw a circle at the point coordinates on the black screen
            cv2.circle(frame, (int(x), int(y)), 5, color, -1)  # Use the unique color

    frame = grid.plot(frame)

    # Display the black screen with points
    cv2.imshow('Points', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='UNAV - Master en Big Data Science - Trabajo Final de Master')
    
    # Add command-line arguments
    parser.add_argument('--config', type=str, default='config.ini', help='Configuration File')
    
    # Parse command-line arguments
    args = parser.parse_args()

    # Call main function with command-line arguments
    main(args.config)