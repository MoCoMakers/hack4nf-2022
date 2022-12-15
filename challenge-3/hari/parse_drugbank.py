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
    row['name'] = drug.findtext(ns + "name")
    row['description'] = drug.findtext(ns + "description")
    row['toxicity'] = drug.findtext(ns + "toxicity")
    

    row['synonyms'] = [synonym.text for synonym in
        drug.findall("{ns}synonyms/{ns}synonym".format(ns = ns))]
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
        drug.findall("{ns}drug-interactions/{ns}drug-interaction/{ns}value".format(ns = ns))
    }
    row['interactions'] = sorted(interactions)

    # Add drug aliases
    aliases = {
        elem.text for elem in 
        drug.findall("{ns}international-brands/{ns}international-brand".format(ns = ns)) +
        drug.findall("{ns}synonyms/{ns}synonym[@language='English']".format(ns = ns)) +
        drug.findall("{ns}international-brands/{ns}international-brand".format(ns = ns)) +
        drug.findall("{ns}products/{ns}product/{ns}name".format(ns = ns))
    }
    aliases.add(row['name'])
    row['aliases'] = sorted(aliases)

    rows.append(row)

def collapse_list_values(row):
    for key, value in row.items():
        if isinstance(value, list):
            row[key] = '|'.join(value)
    return row

rows = list(map(collapse_list_values, rows))

columns = ['drugbank_id', 'name', 'synonyms', 'description', 'type', 'groups', 'toxicity','categories', 'interactions']

drugbank_df = pandas.DataFrame.from_dict(rows)[columns]
#drugbank_df.head()
#nfDrugs = ['DB12001', 'DB11689','DB00227','DB00958','DB08911','DB01229']
#drugbank_df.query('drugbank_id in @nfDrugs')

nfDrugNames = ['Selumetinib', 'Everolimus', 'Lovastatin', 'Pirfenidone', 'Gleevec', 'Trametinib', 'Acetylcysteine amide']
drugbank_df.query('name in @nfDrugNames').to_csv('nFdrugsFile.csv')

