import configparser
from pathlib import Path
from loguru import logger


BASE_PATH = Path(__file__).parent.absolute()
logger.info(f"BASE_PATH={BASE_PATH}")

CONFIG_PATH = BASE_PATH / "config.ini"
logger.info(f"CONFIG_PATH={CONFIG_PATH}")

config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read(CONFIG_PATH)

logger.info("config['Paths']={}".format(list(config["Paths"].items())))
