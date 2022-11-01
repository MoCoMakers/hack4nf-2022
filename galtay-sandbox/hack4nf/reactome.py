"""
[{'dbId': 6802412,
  'displayName': 'neurofibromatosis',
  'databaseName': 'DOID',
  'identifier': '8712',
  'name': ['neurofibromatosis'],
  'synonym': ['Neurofibromatosis 1',
   'neurofibromatosis type IV',
   'Acoustic neurofibromatosis',
   'Recklinghausen&apos;s neurofibromatosis',
   'neurofibromatosis type 4',
   'central Neurofibromatosis',
   'type IV neurofibromatosis of riccardi',
   'neurofibromatosis type 1',
   'neurofibromatosis type 2',
   'peripheral Neurofibromatosis',
   'von Reklinghausen disease'],
  'url': 'https://www.ebi.ac.uk/ols/ontologies/doid/terms?obo_id=DOID:8712',
  'className': 'Disease',
  'schemaClass': 'Disease'}]



"""

RAS_NF1 = "R-HSA-6802953"
NF_DOID = "8712"
NF_dbId = 6802412

import reactome2py.content


discover = reactome2py.content.discover(id=RAS_NF1)
diseases = reactome2py.content.disease()
nf_disease = [el for el in diseases if el['dbId']==NF_dbId][0]
#nf_ent_complx = reactome2py.content.entities_complex(id=RAS_NF1, exclude_structures=False)
