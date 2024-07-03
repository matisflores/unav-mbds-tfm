import sqlite3

class DB:
    _instance = None
    _db = None

    def __init__(self, path:str = None) -> None:
        if path is None:
            print("DB path cannot be None")
            exit(0)

        self._db = sqlite3.connect(path)
        
        c = self._db.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS tracks (tracker TEXT, position TEXT, direction TEXT, cell INTEGER, frame INTEGER, roi TEXT, score REAL, 'timestamp' INTEGER)")
        c.execute("CREATE TABLE IF NOT EXISTS metrics (metric TEXT, value TEXT, frame INTEGER, 'timestamp' DATETIME DEFAULT CURRENT_TIMESTAMP)")
        self._db.commit()
    
    def __del__(self):
        self._db.close()
    
    def save_track(self, tracker, position, direction, cell, frame, roi, score, timestamp):
        c = self._db.cursor()
        c.execute("INSERT INTO tracks (tracker, position, direction, cell, frame, roi, score, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (tracker, str(position), str(direction), cell, frame, roi, score, timestamp))
        self._db.commit()

    def save_metrics(self, metric, frame, value):
        c = self._db.cursor()
        c.execute("INSERT INTO metrics (metric, frame, value) VALUES (?, ?, ?)", (metric, frame, value))
        self._db.commit()

    def load_cell_scores(self):
        c = self._db.cursor()
        c.execute("SELECT cell, avg(score) as score FROM tracks WHERE score > 0 GROUP BY cell")

        return [row for row in c.fetchall()]
    
    def load_cell_qty(self):
        c = self._db.cursor()
        c.execute("SELECT cell, count(1) as qty FROM tracks GROUP BY cell")

        return [row for row in c.fetchall()]
    
    def load_tracking_duration(self):
        c = self._db.cursor()

        c.execute("""
            SELECT 
                tracker,
                MIN(timestamp) AS min_timestamp,
                MAX(timestamp) AS max_timestamp,
                (MAX(timestamp) - MIN(timestamp)) AS timestamp_difference
            FROM 
                tracks
            GROUP BY 
                tracker;
        """)

        return [row for row in c.fetchall()]


    def load_tracking_ids(self):
        c = self._db.cursor()

        # Execute query to fetch distinct tracking IDs
        c.execute("SELECT DISTINCT tracker FROM roi_tracks")

        return [row[0] for row in c.fetchall()]
    
    def load_tracking_points(self, tracker):
        c = self._db.cursor()

        # Retrieve points for the current tracking ID
        c.execute("SELECT * FROM roi_tracks WHERE tracker=? ORDER BY timestamp", (tracker,))
        return c.fetchall()
