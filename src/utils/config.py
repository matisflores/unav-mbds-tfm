import configparser

GENERAL='general'

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

    def get(self, option: str, type = str, section: str = GENERAL):
        return type(self._config.get(section, option))
