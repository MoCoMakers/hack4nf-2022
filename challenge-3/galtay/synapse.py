"""
https://python-docs.synapse.org/build/html/index.html
"""
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

from loguru import logger
import synapseclient
import synapseutils

from config import config

def _read_secrets(
    secrets_path: str = config["Paths"]["SECRETS_PATH"],
) -> Dict[str, str]:
    return json.load(open(secrets_path, "r"))


def get_client(silent=True) -> synapseclient.Synapse:
    secrets = _read_secrets()
    return synapseclient.login(authToken=secrets["SYNAPSE_TOKEN"], silent=silent)


def sync_datasets(
    dataset_synids: Iterable[str],
    sync_path: Union[str, Path] = config["Paths"]["SYNAPSE_PATH"],
) -> List[synapseclient.entity.File]:
    syn = get_client()
    files = []
    for dataset_synid in dataset_synids:
        files.extend(
            synapseutils.syncFromSynapse(
                syn,
                dataset_synid,
                path=Path(sync_path) / dataset_synid,
            )
        )
    return files


if __name__ == "__main__":
    logger.info("syncing synapse datasets")
    files = sync_datasets(["syn5522627"])
