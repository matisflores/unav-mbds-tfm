import argparse
import glob
from io import BytesIO
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

def frame_cell_scores(db_file: str, frame, grid: Grid, color):
    db = DB(db_file)
    cell_scores = db.load_cell_scores()
    cell_scores = {cell[0]: cell[1] for cell in cell_scores}
    grid._cells = [(cell[0],cell[1],cell[2],cell[3],cell_scores.get(cell[3],0)) for cell in grid._cells]

    blue = np.full((grid.cell_size,grid.cell_size,3), color, np.uint8)
    frame_scores = frame.copy()
    for x, y, img, _, score in grid._cells:
        frame_scores[y:y+grid.cell_size, x:x+grid.cell_size] = cv2.addWeighted(img, 1, blue, score, 0.0)

    return frame_scores

def frame_cell_traffic(db_file, frame, grid: Grid, color):
    db = DB(db_file)
    cell_qty = db.load_cell_qty()
    cell_qty = {cell[0]: cell[1] for cell in cell_qty}

    scale = 1.0/max(cell_qty.values())
    for k in cell_qty:
        cell_qty[k] = cell_qty[k] * scale
    
    grid._cells = [(cell[0],cell[1],cell[2],cell[3],cell_qty.get(cell[3],0)) for cell in grid._cells]

    green = np.full((grid.cell_size,grid.cell_size,3), color, np.uint8)
    frame_qty = frame.copy()
    for x, y, img, _, score in grid._cells:
        frame_qty[y:y+grid.cell_size, x:x+grid.cell_size] = cv2.addWeighted(img, 1, green, score, 0.0)

    return frame_qty

def stats_tracking_duration(db_file):
    def timestamp_diff_to_sec(timestamp_difference):
        return int(cv2.getTickFrequency() / timestamp_difference) if timestamp_difference != 0 else 0


    buffer = BytesIO()
    db = DB(db_file)
    results = db.load_tracking_duration()

    results = [(row[0], row[1], row[2], timestamp_diff_to_sec(row[3])) for row in results]

    '''
    # Print the formatted results
    for row in results:
        tracker, min_timestamp, max_timestamp, timestamp_difference = row
        print(f"Tracker: {tracker}")
        print(f"Minimum Timestamp: {min_timestamp}")
        print(f"Maximum Timestamp: {max_timestamp}")
        print(f"Timestamp Difference (seconds): {timestamp_difference}")
        print()
    '''

    # Fetch the timestamp differences
    timestamp_differences = [row[3] for row in results]

    # Plot the distribution
    #plt.hist(timestamp_differences, color='skyblue', edgecolor='black')
    plt.boxplot(timestamp_differences)
    plt.title('Distribution of Trackers duration')
    plt.xlabel('Timestamp Difference (seconds)')
    plt.ylabel('Frequency')
    plt.grid(True)
    #plt.show()
    plt.savefig(buffer, format='png')
    plt.clf()
    buffer.seek(0)
    time_series_img = np.asarray(bytearray(buffer.read()), dtype=np.uint8)
    return cv2.imdecode(time_series_img, cv2.IMREAD_COLOR)

def main(config_file, db_file):
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
    # Load configurations
    config = Config()
    config.load(config_file)

    # Load video
    video = Video(config.source)
    video.open()
    frame = video.read(skip=0)

    # Divide frame
    cell_size = int(config.cell_size)
    grid = Grid(cell_size)
    grid.divide(frame)

    #frame = grid.plot(frame)

    cv2.imwrite(re.sub('\.db', '_cell_scores.jpg', db_file), frame_cell_scores(db_file, frame, grid, (50,0,0)))
    #cv2.imshow('Points', frame_scores)

    cv2.imwrite(re.sub('\.db', '_cell_qty.jpg', db_file), frame_cell_traffic(db_file, frame, grid, (0,50,0)))
    #cv2.imshow('Points', frame_qty)

    # Display the black screen with points
    #cv2.imshow('Points', frame)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    video.release()


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