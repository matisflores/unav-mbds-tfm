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
        c.execute("CREATE TABLE IF NOT EXISTS tracks (tracker TEXT, position TEXT, direction TEXT, cell TEXT, step TEXT, roi TEXT, score TEXT, 'timestamp' DATETIME DEFAULT CURRENT_TIMESTAMP)")
        c.execute("CREATE TABLE IF NOT EXISTS metrics (metric TEXT, value TEXT, step TEXT, 'timestamp' DATETIME DEFAULT CURRENT_TIMESTAMP)")
        self._db.commit()
    
    def __del__(self):
        self._db.close()
    
    def save_track(self, tracker, position, direction, cell, step, roi, score):
        c = self._db.cursor()
        c.execute("INSERT INTO tracks (tracker, position, direction, cell, step, roi, score) VALUES (?, ?, ?, ?, ?, ?, ?)", (tracker, str(position), str(direction), cell, step, roi, score))
        self._db.commit()

    def save_metrics(self, metric, step, value):
        c = self._db.cursor()
        c.execute("INSERT INTO metrics (metric, step, value) VALUES (?, ?, ?)", (metric, step, value))
        self._db.commit()

    def load_cell_scores(self):
        c = self._db.cursor()
        c.execute("SELECT cell, avg(score) as score FROM tracks WHERE score > 0 GROUP BY cell")

        return [row for row in c.fetchall()]
    
    def load_cell_qty(self):
        c = self._db.cursor()
        c.execute("SELECT cell, count(1) as qty FROM tracks GROUP BY cell")

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
