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
import openpyxl

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

config_file_path = "sas_config.toml"
xml_file = "full_database.xml"
supplemental_drugs_file = "supplemental_drugs.csv"


def load_drugbank_values(xml_file):
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
                         drug.findall("{ns}groups/{ns}group".format(ns=ns))]
        row['atc_codes'] = [code.get('code') for code in
                            drug.findall("{ns}atc-codes/{ns}atc-code".format(ns=ns))]
        row['categories'] = [x.findtext(ns + 'category') for x in
                             drug.findall("{ns}categories/{ns}category".format(ns=ns))]

        row['inchi'] = drug.findtext(inchi_template.format(ns=ns))
        row['inchikey'] = drug.findtext(inchikey_template.format(ns=ns))

        # Add drug interactions
        interactions = {
            elem.text for elem in
            drug.findall("{ns}drug-interactions/{ns}drug-interaction/{ns}name".format(ns=ns)) +
            drug.findall("{ns}drug-interactions/{ns}drug-interaction/{ns}description".format(ns=ns))
        }
        row['interactions'] = sorted(interactions)

        # Add Targets
        # Add Targets
        targets = {
            elem.text for elem in
            drug.findall("{ns}targets/{ns}target/{ns}name".format(ns=ns)) +
            drug.findall("{ns}targets/{ns}target/{ns}actions/{ns}action".format(ns=ns))
        }
        row['targets'] = sorted(targets)

        # Add drug aliases
        aliases = {
            elem.text for elem in
            drug.findall("{ns}international-brands/{ns}international-brand".format(ns=ns)) +
            drug.findall("{ns}synonyms/{ns}synonym[@language='English']".format(ns=ns)) +
            drug.findall("{ns}international-brands/{ns}international-brand".format(ns=ns)) +
            drug.findall("{ns}products/{ns}product/{ns}name".format(ns=ns))
        }
        aliases.add(row['drugbank_name'])
        row['aliases'] = sorted(aliases)

        rows.append(row)
    return rows


def collapse_list_values(row):
    for key, value in row.items():
        if isinstance(value, list):
            row[key] = '|'.join(value)
    return row


def rows_to_df(rows):
    rows = list(map(collapse_list_values, rows))

    columns = ['drugbank_id', 'drugbank_name', 'aliases', 'description', \
               'type', 'mechanism-of-action', 'targets', 'groups', 'toxicity', \
               'categories', 'interactions']
    drugbank_df = pandas.DataFrame.from_dict(rows)[columns]

    return drugbank_df


def run_drugbank_on_sas(config_file_path, drugbank_df):
    with open(config_file_path, "r") as fp:
        config = toml.load(fp)

    df_drugs, df_missing, df_all, df_report = sas.main(config)
    synapse_nf_drugs = df_drugs['name']
    matches, drugbank_df = run_df(nf_drugs=synapse_nf_drugs, drugbank_df=drugbank_df)
    return matches, drugbank_df


def run_df(nf_drugs, drugbank_df):
    drugbank_df['aliases_set'] = drugbank_df['aliases'].apply(lambda x: set(x.split('|')))
    drugbank_df['ncgc_id'] = None

    for synapse_nf_drug in nf_drugs:
        mask = drugbank_df['aliases_set'].apply(lambda x: synapse_nf_drug in x)
        drugbank_df.loc[mask, 'ncgc_id'] = 1234

    num_matching_drugs = len(drugbank_df.loc[drugbank_df['ncgc_id'] == 1234])

    matches = drugbank_df.loc[drugbank_df['ncgc_id'] == 1234]
    matches.to_excel('NF-subset-drugbank-matches.xlsx')

    # Save to CSV
    # drop aliases_set column from drugbank_df
    drugbank_df = drugbank_df.drop(columns=['aliases_set'])
    # drugbank_df.to_csv('NF-subset-drugbank.csv', index=False, encoding="utf-8")
    drugbank_df.to_excel('NF-subset-drugbank.xlsx')

    print('Number of synapse-hts datasets drugs matching in drugbank: ', num_matching_drugs)

    return matches, drugbank_df


def run_drugbank_on_supplemental_drugs(drugs_file, drugbank_df):
    df_supplement = pd.read_csv(drugs_file)
    supplemental_nf_drugs = df_supplement['name']
    matches, drugbank_df = run_df(nf_drugs=supplemental_nf_drugs, drugbank_df=drugbank_df)
    return matches, drugbank_df


def include_alternative_drugs(matches):
    logger.info("Inserting `alternative_approved_drugs` column into the matches dataframe...")
    matches["alternative_approved_drugs"] = matches.apply(lambda row: "|".join(matches.loc[(row['targets'] == matches['targets']) & \
                                          (row['drugbank_name'] != matches['drugbank_name']) & \
                                           (matches["groups"].str.contains("approved")), 'drugbank_name'].to_list()),
                                           axis=1
                                        )
    logger.info("Column insertion done")
    logger.info("Properties of column `alternative_approved_drugs`:")
    matches_backupfile = 'NF-drugbank-matches-with-alternatives.xlsx'
    logger.info("Creating backup Excel sheet - {file}...".format(file=matches_backupfile))
    matches.to_excel(matches_backupfile)
    logger.info("Backup done. {file} created".format(file=matches_backupfile))


def get_drugname_by_target(target, drugbank_df):
    for dbid in drugbank_df['drugbank_id'].to_list():
        found_df = drugbank_df.loc[drugbank_df['drugbank_id'] == dbid]
        found_targets = found_df['targets']
        print(str(found_df['drugbank_name']) + ": "+str(found_targets.to_list()))
def extend_matches(matches, drugbank_df):
    for match in matches['drugbank_name'].tolist():
        print(match)
        #match_df = drugbank_df.loc[drugbank_df['drugbank_name'] == match]
        initial_targets = matches.loc[matches['drugbank_name'] == match]
        print(initial_targets)
        match_df = drugbank_df.loc[drugbank_df['targets'].isin(initial_targets)]
        names_df = match_df['drugbank_name']
        names_dict = names_df.to_dict()
        print(names_dict)
    #
    #     mask = drugbank_df['targets'].isin(matches['targets'])
    #     drugbank_df['aliases_set'].apply(lambda x: matches['targets'].isin())
    #     drugbank_df.loc[mask, 'collider_id'] = 1234
    #     matches = drugbank_df.loc[drugbank_df['collider_id'] == 1234]
    #
    #     colliding_matches = drugbank_df.loc[]
    #     matches['colliders'] = drugbank_df['targets'].apply(lambda x: set(
    #         x.split('|')
    #     ))
    #
    # colliding_matches.to_excel('colliding-matches.xlsx')


if __name__ == "__main__":
    if not os.path.exists('drugbank_df.pkl'):
        rows = load_drugbank_values(xml_file)
        drugbank_df = rows_to_df(rows)
        drugbank_df.to_pickle("drugbank_df.pkl")
    else:
        drugbank_df = pd.read_pickle("drugbank_df.pkl")

    if not os.path.exists('matches_df.pkl'):
        # matches, drugbank_df = run_drugbank_on_sas(config_file_path, drugbank_df)
        matches, drugbank_df = run_drugbank_on_supplemental_drugs(supplemental_drugs_file, drugbank_df)
        matches.to_pickle("matches_df.pkl")
    else:
        matches = pd.read_pickle('matches_df.pkl')
    #extend_matches(matches, drugbank_df)
    target='Auranofin'
    get_drugname_by_target(target, drugbank_df)
    include_alternative_drugs(matches)
    print("Done")
