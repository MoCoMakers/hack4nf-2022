"""
https://python-docs.synapse.org/build/html/index.html
"""
import json
import os
from pathlib import Path

import synapseclient
import synapseutils


GENIE_12 = "genie-12.0-public"
GENIE_13 = "genie-13.3-consortium"


DATA_CLINICAL_PATIENT = "data_clinical_patient"
DATA_CLINICAL_SAMPLE = "data_clinical_sample"
DATA_MUTATIONS_EXTENDED = "data_mutations_extended"
DATA_CNA = "data_CNA"
DATA_CNA_HG19_SEG = "data_cna_hg19"


# set environment variable to something like
# /home/galtay/data/hack4nf/synapse
# if not set it will use the default SYNAPSE CACHE
SYNC_PATH = os.getenv("HACK4NF_SYNAPSE_SYNC_PATH")

# path to secrets.json file
SECRETS_PATH = os.getenv("HACK4NF_SYNAPSE_SECRETS_PATH")


DATASET_NAME_TO_SYNID = {
    GENIE_12: "syn32309524",
    GENIE_13: "syn36709873",
}

FILE_NAME_TO_SYNID = {
    GENIE_12: {
        DATA_CLINICAL_PATIENT: "syn32689054",
        DATA_CLINICAL_SAMPLE: "syn32689057",
        DATA_MUTATIONS_EXTENDED: "syn32689317",
        DATA_CNA: "syn32689019",
        DATA_CNA_HG19_SEG: "syn32689379",
    },
    GENIE_13: {
        DATA_CLINICAL_PATIENT: "syn36710136",
        DATA_CLINICAL_SAMPLE: "syn36710137",
        DATA_MUTATIONS_EXTENDED: "syn36710142",
        DATA_CNA: "syn36710134",
        DATA_CNA_HG19_SEG: "syn36710143",
    },
}


def get_file_name_to_path(sync_path=SYNC_PATH):

    genie_12_path = Path(sync_path) / DATASET_NAME_TO_SYNID[GENIE_12]
    genie_13_path = Path(sync_path) / DATASET_NAME_TO_SYNID[GENIE_13]

    file_name_to_path = {
        GENIE_12: {
            DATA_CLINICAL_PATIENT: genie_12_path / f"{DATA_CLINICAL_PATIENT}.txt",
            DATA_CLINICAL_SAMPLE: genie_12_path / f"{DATA_CLINICAL_SAMPLE}.txt",
            DATA_MUTATIONS_EXTENDED: genie_12_path / f"{DATA_MUTATIONS_EXTENDED}.txt",
            DATA_CNA: genie_12_path / f"{DATA_CNA}.txt",
            DATA_CNA_HG19_SEG: genie_12_path / f"genie_{DATA_CNA_HG19_SEG}.seg",
        },
        GENIE_13: {
            DATA_CLINICAL_PATIENT: genie_13_path / f"{DATA_CLINICAL_PATIENT}_13.3-consortium.txt",
            DATA_CLINICAL_SAMPLE: genie_13_path / f"{DATA_CLINICAL_SAMPLE}_13.3-consortium.txt",
            DATA_MUTATIONS_EXTENDED: genie_13_path / f"{DATA_MUTATIONS_EXTENDED}_13.3-consortium.txt",
            DATA_CNA: genie_13_path / f"{DATA_CNA}_13.3-consortium.txt",
            DATA_CNA_HG19_SEG: genie_13_path / f"genie_private_{DATA_CNA_HG19_SEG}_13.3-consortium.seg",
        },
    }
    return file_name_to_path


def _read_secrets():
    if SECRETS_PATH is None:
        raise ValueError("please set environment variable HACK4NF_SYNAPSE_SECRETS_PATH")
    return json.load(open(SECRETS_PATH, "r"))


def get_client(silent=True):
    secrets = _read_secrets()
    return synapseclient.login(
        authToken=secrets["SYNAPSE_TOKEN"],
        silent=silent)


def sync_datasets(dataset_synids=None):
    if dataset_synids is None:
        dataset_synids = DATASET_NAME_TO_SYNID.values()
    syn = get_client()
    files = []
    for dataset_synid in dataset_synids:
        files.extend(synapseutils.syncFromSynapse(
            syn,
            dataset_synid,
            path = Path(SYNC_PATH) / dataset_synid,
        ))
    return files


if __name__ == "__main__":

    sync_datasets()
