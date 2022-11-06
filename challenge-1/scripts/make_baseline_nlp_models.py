import json
import os
import pandas as pd

import plotly.express as px

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from nextgenlp import synapse
from nextgenlp import genie
from nextgenlp import genie_constants
from nextgenlp import embedders


#GENIE_VERSION = "genie-12.0-public"
GENIE_VERSION = "genie-13.3-consortium"
SYNC_PATH = synapse.SYNC_PATH
EMBEDDINGS_PATH = os.path.join(SYNC_PATH, "../embeddings")

#Y_PREDICT = 'ONCOTREE_CODE'
#Y_PREDICT = "CANCER_TYPE_DETAILED"
Y_PREDICT = "CANCER_TYPE"

KFOLDS = 5
MIN_Y_COUNT = KFOLDS * 6

#GENIE_SUBSET = "MSK-IMPACT468"
#GENIE_SUBSET = "MSK-NOHEME"
#GENIE_SUBSET = "DFCI-MSK-UCSF"
GENIE_SUBSET = "UCSF"
#GENIE_SUBSET = "ALL"

SENT_KEY = "mut_sent"


def get_count_vectorizer(**cv_kwargs):
    """Get a count vectorizer appropriate for pre-tokenized text"""
    return CountVectorizer(
        tokenizer=lambda x: x, preprocessor=lambda x: x, token_pattern=None, **cv_kwargs
    )


syn_file_paths = synapse.get_file_name_to_path(genie_version=GENIE_VERSION)
syn_file_paths


gds = {}
gds["ALL"] = genie.GenieData.from_file_paths(syn_file_paths)
gds[GENIE_SUBSET] = gds["ALL"].subset_from_seq_assay_id_group(GENIE_SUBSET)
gd = gds[GENIE_SUBSET].subset_from_ycol(Y_PREDICT, MIN_Y_COUNT)


df_train, df_test = train_test_split(gd.df_dcs, stratify=gd.df_dcs[Y_PREDICT])

pipe = Pipeline([
    ("count", get_count_vectorizer()),
    ("clf", LogisticRegression()),
])

pipe.fit(df_train[SENT_KEY], df_train[Y_PREDICT])
y_pred = pipe.predict(df_test[SENT_KEY])
cls_report_dict = classification_report(
    df_test[Y_PREDICT], y_pred, output_dict=True
)
df_clf_report = (
    pd.DataFrame(cls_report_dict)
    .drop(columns=["accuracy", "macro avg", "weighted avg"])
    .T
)


feature_names = pipe.named_steps["count"].get_feature_names()
classes = pipe.named_steps['clf'].classes_
class_name_to_index = {name: ii for ii, name in enumerate(classes)}
coefs = pipe.named_steps['clf'].coef_

df_clf_report_good = df_clf_report[df_clf_report['f1-score']>0.5].sort_values('f1-score', ascending=False)

for class_name, row in df_clf_report_good.iterrows():
    class_index = class_name_to_index[class_name]
    feature_weights = sorted(list(zip(coefs[class_index], feature_names)), reverse=True)

    print("class_name: {}, f1={}".format(class_name, row['f1-score']))
    for weight, gene in feature_weights[:10]:
        print(f"gene={gene}, feature_weight={weight:.2f}")
    print()
