import argparse
import glob
import os
import cv2
import re
from datetime import datetime
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from utils.Grid import Grid
from utils.Video import Video
from utils.Config import Config
from utils.DB import DB

def colorFader(value):
    c1 = 'red'
    c2 = 'green'
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    return (1-value)*c1 + value*c2

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

def main(config_file, db_file):
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
    db = DB(db_file)

    # Load video
    video = Video(config.source)
    video.open()
    frame = video.read(skip=0)

    # Divide frame
    cell_size = int(config.cell_size)
    grid = Grid(cell_size)
    grid.divide(frame)

    #cell_score
    cell_scores = db.load_cell_scores()
    cell_scores = {cell[0]: cell[1] for cell in cell_scores}
    grid._cells = [(cell[0],cell[1],cell[2],cell[3],cell_scores.get(str(cell[3]),0)) for cell in grid._cells]

    #frame = grid.plot(frame)
    blue = np.full((grid.cell_size,grid.cell_size,3), (50,0,0), np.uint8)
    frame_scores = frame.copy()
    for x, y, img, _, score in grid._cells:
        frame_scores[y:y+grid.cell_size, x:x+grid.cell_size] = cv2.addWeighted(img, 1, blue, score, 0.0)

    cv2.imwrite(re.sub('\.db', '_cell_scores.jpg', db_file), frame_scores)
    #cv2.imshow('Points', frame_scores)


    #cell_qty
    cell_qty = db.load_cell_qty()
    cell_qty = {cell[0]: cell[1] for cell in cell_qty}

    factor=1.0/max(cell_qty.values())
    for k in cell_qty:
        cell_qty[k] = cell_qty[k]*factor
    
    grid._cells = [(cell[0],cell[1],cell[2],cell[3],cell_qty.get(str(cell[3]),0)) for cell in grid._cells]

    #frame = grid.plot(frame)
    green = np.full((grid.cell_size,grid.cell_size,3), (0,50,0), np.uint8)
    frame_qty = frame.copy()
    for x, y, img, _, score in grid._cells:
        frame_qty[y:y+grid.cell_size, x:x+grid.cell_size] = cv2.addWeighted(img, 1, green, score, 0.0)

    cv2.imwrite(re.sub('\.db', '_cell_qty.jpg', db_file), frame_qty)
    #cv2.imshow('Points', frame_qty)


    '''
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
    '''

    # Display the black screen with points
    #cv2.imshow('Points', frame)
    #cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='UNAV - Master en Big Data Science - Trabajo Final de Master')
    
    # Add command-line arguments
    parser.add_argument('--config', type=str, default='config.ini', help='Configuration File')
    parser.add_argument('--database', type=str, default='', help='Database File')
    
    # Parse command-line arguments
    args = parser.parse_args()

    if args.database == '':
        config = Config()
        config.load(args.config)
        list_of_files = glob.glob(config.data_dir + '/' + os.path.basename(config.source) + '*.db') # * means all if need specific format then *.csv
        args.database = max(list_of_files, key=os.path.getctime)

    # Call main function with command-line arguments
    main(args.config, args.database)