"""
https://python-docs.synapse.org/build/html/index.html
"""
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from loguru import logger
import synapseclient
import synapseutils


GENIE_12 = "genie-12.0-public"
GENIE_13 = "genie-13.3-consortium"
VALID_GENIE_VERSIONS = [GENIE_12, GENIE_13]


ASSAY_INFORMATION = "assay_information"
DATA_CLINICAL_PATIENT = "data_clinical_patient"
DATA_CLINICAL_SAMPLE = "data_clinical_sample"
DATA_FUSIONS = "data_fusions"
DATA_GENE_MATRIX = "data_gene_matrix"
DATA_MUTATIONS_EXTENDED = "data_mutations_extended"
DATA_CNA = "data_CNA"
DATA_CNA_HG19_SEG = "data_cna_hg19"
GENOMIC_INFORMATION = "genomic_information"


# set environment variable to something like
# /home/galtay/data/hack4nf/synapse
# if not set it will use the default SYNAPSE CACHE
SYNC_PATH = os.getenv("HACK4NF_SYNAPSE_SYNC_PATH")
logger.info(f"SYNC_PATH={SYNC_PATH}")

# path to secrets.json file
SECRETS_PATH = os.getenv("HACK4NF_SYNAPSE_SECRETS_PATH")
logger.info(f"SECRETS_PATH={SECRETS_PATH}")


# these are used to download the entire datasets
DATASET_NAME_TO_SYNID = {
    GENIE_12: "syn32309524",
    GENIE_13: "syn36709873",
}

FILE_NAME_TO_SYNID = {
    GENIE_12: {
        ASSAY_INFORMATION: "syn32688743",
        DATA_CLINICAL_PATIENT: "syn32689054",
        DATA_CLINICAL_SAMPLE: "syn32689057",
        DATA_FUSIONS: "syn32689059",
        DATA_GENE_MATRIX: "syn32689060",
        DATA_MUTATIONS_EXTENDED: "syn32689317",
        DATA_CNA: "syn32689019",
        DATA_CNA_HG19_SEG: "syn32689379",
        GENOMIC_INFORMATION: "syn32690864",
    },
    GENIE_13: {
        ASSAY_INFORMATION: "syn36710133",
        DATA_CLINICAL_PATIENT: "syn36710136",
        DATA_CLINICAL_SAMPLE: "syn36710137",
        DATA_FUSIONS: "syn36710139",
        DATA_GENE_MATRIX: "syn36710140",
        DATA_MUTATIONS_EXTENDED: "syn36710142",
        DATA_CNA: "syn36710134",
        DATA_CNA_HG19_SEG: "syn36710143",
        GENOMIC_INFORMATION: "syn36710146",
    },
}


def get_file_name_to_path(
    sync_path: str = SYNC_PATH,
    genie_version: str = "genie-13.3-consortium",
) -> Dict[str, Path]:

    """Return the paths to GENIE datasets."""

    if genie_version not in VALID_GENIE_VERSIONS:
        raise ValueError(f"genie version must be one of {VALID_GENIE_VERSIONS}")

    genie_path = Path(sync_path) / DATASET_NAME_TO_SYNID[genie_version]
    logger.info(f"genie_path={genie_path}")

    if genie_version == GENIE_12:
        file_name_to_path = {
            ASSAY_INFORMATION: genie_path / f"{ASSAY_INFORMATION}.txt",
            DATA_CLINICAL_PATIENT: genie_path / f"{DATA_CLINICAL_PATIENT}.txt",
            DATA_CLINICAL_SAMPLE: genie_path / f"{DATA_CLINICAL_SAMPLE}.txt",
            DATA_FUSIONS: genie_path / f"{DATA_FUSIONS}.txt",
            DATA_GENE_MATRIX: genie_path / f"{DATA_GENE_MATRIX}.txt",
            DATA_MUTATIONS_EXTENDED: genie_path / f"{DATA_MUTATIONS_EXTENDED}.txt",
            DATA_CNA: genie_path / f"{DATA_CNA}.txt",
            DATA_CNA_HG19_SEG: genie_path / f"genie_{DATA_CNA_HG19_SEG}.seg",
            GENOMIC_INFORMATION: genie_path / f"{GENOMIC_INFORMATION}.txt",
        }

    if genie_version == GENIE_13:
        file_name_to_path = {
            ASSAY_INFORMATION: genie_path / f"{ASSAY_INFORMATION}_13.3-consortium.txt",
            DATA_CLINICAL_PATIENT: genie_path
            / f"{DATA_CLINICAL_PATIENT}_13.3-consortium.txt",
            DATA_CLINICAL_SAMPLE: genie_path
            / f"{DATA_CLINICAL_SAMPLE}_13.3-consortium.txt",
            DATA_FUSIONS: genie_path / f"{DATA_FUSIONS}_13.3-consortium.txt",
            DATA_GENE_MATRIX: genie_path / f"{DATA_GENE_MATRIX}_13.3-consortium.txt",
            DATA_MUTATIONS_EXTENDED: genie_path
            / f"{DATA_MUTATIONS_EXTENDED}_13.3-consortium.txt",
            DATA_CNA: genie_path / f"{DATA_CNA}_13.3-consortium.txt",
            DATA_CNA_HG19_SEG: genie_path
            / f"genie_private_{DATA_CNA_HG19_SEG}_13.3-consortium.seg",
            GENOMIC_INFORMATION: genie_path
            / f"{GENOMIC_INFORMATION}_13.3-consortium.txt",
        }

    return file_name_to_path


def _read_secrets() -> Dict[str, str]:
    if SECRETS_PATH is None:
        raise ValueError("please set environment variable HACK4NF_SYNAPSE_SECRETS_PATH")
    return json.load(open(SECRETS_PATH, "r"))


def get_client(silent=True) -> synapseclient.Synapse:
    secrets = _read_secrets()
    return synapseclient.login(authToken=secrets["SYNAPSE_TOKEN"], silent=silent)


def sync_datasets(
    dataset_synids: Optional[Iterable[str]] = None,
) -> List[synapseclient.entity.File]:
    if dataset_synids is None:
        dataset_synids = DATASET_NAME_TO_SYNID.values()
    syn = get_client()
    files = []
    for dataset_synid in dataset_synids:
        files.extend(
            synapseutils.syncFromSynapse(
                syn,
                dataset_synid,
                path=Path(SYNC_PATH) / dataset_synid,
            )
        )
    return files


if __name__ == "__main__":
    logger.info("syncing all default synapse datasets")
    files = sync_datasets()
