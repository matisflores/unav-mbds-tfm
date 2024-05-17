import sqlite3

class DB:
    _instance = None
    _db = None

    def __new__(cls, path:str):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._db = sqlite3.connect(path)

        return cls._instance
    
    def save_tracker_rois(self, tracker, roi, cell):
        c = self._db.cursor()

        # Create table if not exists
        c.execute("CREATE TABLE IF NOT EXISTS roi_tracks (tracker TEXT, roi TEXT, cell TEXT, 'timestamp' DATETIME DEFAULT CURRENT_TIMESTAMP)")

        # Insert ROI and selected zones into the database
        c.execute("INSERT INTO roi_tracks (tracker, roi, cell) VALUES (?, ?, ?)", (tracker, roi, cell))

        self._db.commit()

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

    def __del__(self):
        self._db.close()