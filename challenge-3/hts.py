"""
https://python-docs.synapse.org/build/html/index.html
"""
from pathlib import Path
from loguru import logger
import pandas as pd
from config import config


synid = "syn5522627"
base_path = Path(config["Paths"]["SYNAPSE_PATH"]) / synid
file_name = "NTAP ipNF02.3 2l MIPE qHTS.csv"
file_path = base_path / file_name
df = pd.read_csv(file_path)
