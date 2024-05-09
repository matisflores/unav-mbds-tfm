import sqlite3
from config.config import Config

class DB:
    _instance = None
    _config = None
    _db = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._config = Config()
            cls._db = sqlite3.connect(cls._config.path_data_dir + '/tracking.db')

        return cls._instance
    
    def save_tracker_rois(self, tracker, rois):
        c = self._db.cursor()

        # Create table if not exists
        c.execute("CREATE TABLE IF NOT EXISTS roi_tracks (tracker TEXT, rois TEXT, 'timestamp' DATETIME DEFAULT CURRENT_TIMESTAMP)")

        # Insert ROI and selected zones into the database
        c.execute("INSERT INTO roi_tracks (tracker, rois) VALUES (?, ?)", (tracker, str(rois)))

        self._db.commit()

    def __del__(self):
        self._db.close()