import configparser
from pathlib import Path

BASE_PATH = Path( __file__ ).parent.parent.absolute()

config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read(BASE_PATH / "config.ini")
