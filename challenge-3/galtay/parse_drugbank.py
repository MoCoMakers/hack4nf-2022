'''
Modified version of dhimmel's drugbank parse github package to filter for Hack4NF 2022 fields of interest
Source: https://github.com/dhimmel/drugbank/blob/gh-pages/parse.ipynb
'''

import os
import csv
import collections
import re
import io
import json
import xml.etree.ElementTree as ET

import requests
import pandas

from collections import Counter
from glob import glob
import os
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import plotly.express as px
import toml

import single_agent_screens as sas

config_file_path = "/Users/Hari_Donthi/Downloads/rosalind/hack4nf-2022/challenge-3/galtay/sas_config.toml"
with open(config_file_path, "r") as fp:
    config = toml.load(fp)

df_drugs, df_missing, df_all, df_report = sas.main(config)

xml_file = "/Users/Hari_Donthi/Downloads/full_database.xml"
tree = ET.parse(xml_file)
root = tree.getroot()

ns = '{http://www.drugbank.ca}'
inchikey_template = "{ns}calculated-properties/{ns}property[{ns}kind='InChIKey']/{ns}value"
inchi_template = "{ns}calculated-properties/{ns}property[{ns}kind='InChI']/{ns}value"

rows = list()
for i, drug in enumerate(root):
    row = collections.OrderedDict()
    assert drug.tag == ns + 'drug'
    row['type'] = drug.get('type')
    row['drugbank_id'] = drug.findtext(ns + "drugbank-id[@primary='true']")
    row['drugbank_name'] = drug.findtext(ns + "name")
    row['description'] = drug.findtext(ns + "description")
    row['toxicity'] = drug.findtext(ns + "toxicity")
    row['mechanism-of-action'] = drug.findtext(ns + "mechanism-of-action")
    
    row['groups'] = [group.text for group in
        drug.findall("{ns}groups/{ns}group".format(ns = ns))]
    row['atc_codes'] = [code.get('code') for code in
        drug.findall("{ns}atc-codes/{ns}atc-code".format(ns = ns))]
    row['categories'] = [x.findtext(ns + 'category') for x in
        drug.findall("{ns}categories/{ns}category".format(ns = ns))]

    
    row['inchi'] = drug.findtext(inchi_template.format(ns = ns))
    row['inchikey'] = drug.findtext(inchikey_template.format(ns = ns))


    # Add drug interactions
    interactions = {
        elem.text for elem in
        drug.findall("{ns}drug-interactions/{ns}drug-interaction/{ns}name".format(ns = ns)) +
        drug.findall("{ns}drug-interactions/{ns}drug-interaction/{ns}description".format(ns = ns))
    }
    row['interactions'] = sorted(interactions)


    #Add Targets
    #Add Targets
    targets = {
        elem.text for elem in
        drug.findall("{ns}targets/{ns}target/{ns}name".format(ns = ns)) + 
        drug.findall("{ns}targets/{ns}target/{ns}actions/{ns}action".format(ns = ns))
        }
    row['targets'] = sorted(targets)

    # Add drug aliases
    aliases = {
        elem.text for elem in 
        drug.findall("{ns}international-brands/{ns}international-brand".format(ns = ns)) +
        drug.findall("{ns}synonyms/{ns}synonym[@language='English']".format(ns = ns)) +
        drug.findall("{ns}international-brands/{ns}international-brand".format(ns = ns)) +
        drug.findall("{ns}products/{ns}product/{ns}name".format(ns = ns))
    }
    aliases.add(row['drugbank_name'])
    row['aliases'] = sorted(aliases)

    rows.append(row)

def collapse_list_values(row):
    for key, value in row.items():
        if isinstance(value, list):
            row[key] = '|'.join(value)
    return row

rows = list(map(collapse_list_values, rows))

columns = ['drugbank_id', 'drugbank_name', 'aliases', 'description',\
        'type', 'mechanism-of-action', 'targets', 'groups', 'toxicity',\
            'categories', 'interactions']

drugbank_df = pandas.DataFrame.from_dict(rows)[columns]

synapse_nf_drugs = df_drugs['name']

drugbank_df['aliases_set'] = drugbank_df['aliases'].apply(lambda x: set(x.split('|')))
drugbank_df['ncgc_id'] = None

for synapse_nf_drug in synapse_nf_drugs:
    mask = drugbank_df['aliases_set'].apply(lambda x: synapse_nf_drug in x)
    drugbank_df.loc[mask,'ncgc_id']=1234

num_matching_drugs = len(drugbank_df.loc[drugbank_df['ncgc_id'] == 1234])

#Save to CSV
#drop aliases_set column from drugbank_df
drugbank_df = drugbank_df.drop(columns=['aliases_set'])
drugbank_df.to_csv('NF-subset-drugbank.csv', index=False)

print('Number of synapse-hts datasets drugs matching in drugbank: ', num_matching_drugs)
