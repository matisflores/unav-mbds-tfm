import configparser

def load_config(filename):
    """
    Load configurations from a file using configparser.

    Args:
        filename (str): The name of the file containing configurations.

    Returns:
        configparser.ConfigParser: A ConfigParser object containing the configurations.
    """
    config = configparser.ConfigParser()
    config.read(filename)
    return config