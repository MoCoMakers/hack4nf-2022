"""
https://www.synapse.org/#!Synapse:syn7222066/wiki/410922
https://python-docs.synapse.org/build/html/index.html#
https://help.synapse.org/docs/
https://www.synapse.org/#!Synapse:syn36709873
"""
import json
import pathlib
import synapseclient
import synapseutils


_PATH_HERE = pathlib.Path(__file__).parent.absolute()


DATASET_IDS = {
    "genie-12.0-public": "syn32309524",
    "genie-13.3-consortium": "syn36709873",
    #"drug-screen-pnf-cell-lines": "syn4939906",
    "drug-screen-pnf-cell-lines-single-agent-screens": "syn5522627",
}


def _read_secrets():
    secrets_path = _PATH_HERE.parent / "secrets.json"
    return json.load(open(secrets_path, "r"))


def get_client(silent=True):
    secrets = _read_secrets()
    return synapseclient.login(
        authToken=secrets["SYNAPSE_TOKEN"],
        silent=silent)


def sync_datasets(dataset_ids=None):
    if dataset_ids is None:
        dataset_ids = DATASET_IDS.values()
    syn = get_client()
    files = []
    for dataset_id in dataset_ids:
        files.extend(synapseutils.syncFromSynapse(syn, dataset_id))
    return files


def get_dataset(dataset_id):
    syn = get_client()
    return syn.get(dataset_id)


if __name__ == "__main__":

    sync_datasets()
