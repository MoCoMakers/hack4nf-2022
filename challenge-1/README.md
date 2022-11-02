# Hack4NF 2022

* https://hack4nf.bemyapp.com/

# Install `nextgenlp` Python Package 

Create and activate a python environment with your favorite tool. 
An example for conda that would be compatible with the default python version on the 
AHA Precision Platform would be, 

```bash
conda create --name ng37 python=3.7
conda activate ng37
```

Run the following command in the directory that contains the `setup.cfg` file. 
You might have to update to the latest version of pip

```bash
pip install -U pip
```

```bash
pip install -e .
```

NOTE: A `requirements.txt` file is included to show one set of package versions that worked on one machine. 
However, the pip install command above should take care of installing all the requirements for this package.


# Setup Data Config

## config.ini 

Copy `config.ini.template` to `config.ini` and edit the line that starts with `DATA_PATH`. 
This should point to an empty directory.
`nextgenlp` will use this location to store synapse datasets and derived data. 

## secrets.json

In order to securely download data from synapse you will need a personal access token. 
Generate one by folling the instructions at the links below, 

* https://help.synapse.org/docs/Client-Configuration.1985446156.html
* https://www.synapse.org/#!PersonalAccessTokens

Next, copy `secrets.json.template` to `DATA_PATH/secrets.json` 
and edit the new file to contain your personal access token. 


# Access to GENIE

## GENIE 12.0

* https://www.synapse.org/#!Synapse:syn32309524

## GENIE 13.3 (special access request required)

* https://www.synapse.org/#!Synapse:syn36709873

Start by requesting access to GENIE dataset 

* https://www.synapse.org/#!Synapse:syn34548529/wiki/618904

After you get a confirmation email - you will need to go back the the site, click "Request Access" and click "Accept" on the electronic terms that pop up. You should then have permission to download the dataset. 

# Downloading from Synapse

Run the `synapse.py` file, 

```bash
python nextgenlp/synapse.py
```

check that the path you specified in `DATA_PATH` is populated. 


# Getting Started 

Launch jupyter notebook and checkout the `genie_explore.ipynb` notebook. 

```bash
jupyter notebook
```



