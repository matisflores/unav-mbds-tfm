import configparser

GENERAL='General'
PATHS='Paths'
OPENCV='Opencv'
YOLO='Yolo'
KALMAN='Kalman'

class Config:
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, file):
        """
        Load configurations from a file using configparser.

        Returns:
            configparser.ConfigParser: A ConfigParser object containing the configurations.
        """
        self._config = configparser.ConfigParser()
        self._config.read(file)

    @property
    def path_data_dir(self):
        return self._config.get(PATHS, 'data_dir')
    
    @property
    def path_assets_dir(self):
        return self._config.get(PATHS, 'assets_dir')
    
    @property
    def oc_window_title(self):
        return self._config.get(OPENCV, 'window_title')
    
    @property
    def cell_size(self):
        return self._config.get(GENERAL, 'cell_size')
    
    @property
    def source(self):
        return self._config.get(GENERAL, 'source')
    
    @property
    def yolo_confidence_threshold(self):
        return self._config.get(YOLO, 'confidence_threshold')
    
    @property
    def kalman_min_iou(self):
        return self._config.get(KALMAN, 'min_iou')
    
    @property
    def kalman_min_steps_alive(self):
        return self._config.get(KALMAN, 'min_steps_alive')