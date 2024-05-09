import configparser

GENERAL='General'

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
        return self._config.get('Paths', 'data_dir')
    
    @property
    def path_assets_dir(self):
        return self._config.get('Paths', 'assets_dir')
    
    @property
    def oc_window_title(self):
        return self._config.get('Opencv', 'window_title')
    
    @property
    def zone_size(self):
        return self._config.get(GENERAL, 'zone_size')
    
    @property
    def source(self):
        return self._config.get(GENERAL, 'source')