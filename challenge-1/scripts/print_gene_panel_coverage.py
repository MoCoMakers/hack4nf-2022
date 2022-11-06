import json
import os
import pandas as pd

from nextgenlp import synapse
from nextgenlp import genie
from nextgenlp import genie_constants
from nextgenlp import embedders


#GENIE_VERSION = "genie-12.0-public"
GENIE_VERSION = "genie-13.3-consortium"
SYNC_PATH = synapse.SYNC_PATH

syn_file_paths = synapse.get_file_name_to_path(genie_version=GENIE_VERSION)

gds = {}
gds["ALL"] = genie.GenieData.from_file_paths(syn_file_paths)


rows = []
for seq_assay_id_group in genie_constants.SEQ_ASSAY_ID_GROUPS.keys():
    if seq_assay_id_group == "ALL":
        continue
    gd = gds["ALL"].subset_from_seq_assay_id_group(seq_assay_id_group)
    gds[seq_assay_id_group] = gd

    row = {
        "subset": gd.seq_assay_id_group,
        "panels": len(gd.seq_assay_ids),
        "genes": len(gd.genes),
        "samples": gd.df_dcs.shape[0],
        "variants": gd.df_mut.shape[0],
    }
    rows.append(row)


df_report = pd.DataFrame.from_records(rows)
