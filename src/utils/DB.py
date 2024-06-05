import sqlite3

class DB:
    _instance = None
    _db = None

    def __new__(cls, path:str = None):
        if cls._instance is not None:
            return cls._instance
        
        if path is None:
            print("DB path cannot be None")
            exit(0)

        cls._instance = super().__new__(cls)
        cls._db = sqlite3.connect(path)
        
        c = cls._db.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS roi_tracks (tracker TEXT, roi TEXT, cell TEXT, step TEXT, 'timestamp' DATETIME DEFAULT CURRENT_TIMESTAMP)")
        c.execute("CREATE TABLE IF NOT EXISTS metrics (metric TEXT, value TEXT, step TEXT, 'timestamp' DATETIME DEFAULT CURRENT_TIMESTAMP)")
        cls._db.commit()

        return cls._instance
    
    def __del__(self):
        self._db.close()
    
    def save_tracker_rois(self, tracker, roi, cell, step):
        c = self._db.cursor()
        #c.execute("CREATE TABLE IF NOT EXISTS roi_tracks (tracker TEXT, roi TEXT, cell TEXT, step TEXT, 'timestamp' DATETIME DEFAULT CURRENT_TIMESTAMP)")
        c.execute("INSERT INTO roi_tracks (tracker, roi, cell, step) VALUES (?, ?, ?, ?)", (tracker, roi, cell, step))
        self._db.commit()

    def save_metrics(self, metric, step, value):
        c = self._db.cursor()
        #c.execute("CREATE TABLE IF NOT EXISTS metrics (metric TEXT, value TEXT, step TEXT, 'timestamp' DATETIME DEFAULT CURRENT_TIMESTAMP)")
        c.execute("INSERT INTO metrics (metric, step, value) VALUES (?, ?, ?)", (metric, step, value))
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
