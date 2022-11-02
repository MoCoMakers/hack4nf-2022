# Hack4NF

2022 Hack4NF https://hack4nf.bemyapp.com/

# Install hack4nf Python Package 

Create a python environment with your favorite python thing (conda, virtual env, ..). 
An example for conda that would be compatible with the default python version on the 
AHA Precision Platform would be, 

```bash
conda create --name hack4nf python=3.7
```

Run the following command in the directory that contains the `setup.cfg` file. 
You might have to update to the latest version of pip

```bash
pip install -U pip
```


```bash
pip install -e .
```

For everything to "just work" you will need to export two environment variables

This one will determine where synapse datasets will be downloaded

```bash
HACK4NF_SYNAPSE_SYNC_PATH=/path/to/data/synapse
```

This one needs to be the full path to your `secrets.json` file 

```bash
HACK4NF_SYNAPSE_SECRETS_PATH=/path/to/secrets.json
```

# Access to GENIE 

Start by requesting access to GENIE dataset - https://www.synapse.org/#!Synapse:syn34548529/wiki/618904

After you get a confirmation email - you will need to go back the the site, click "Request Access" and click Accept on the electronic terms that pop up. You should then have the ability to download from the website (and thus the code).

# downloading from synapse

Synapse.org -> Login (must have account, see hackathon site) -> Generate a code per:

https://help.synapse.org/docs/Client-Configuration.1985446156.html

https://www.synapse.org/#!PersonalAccessTokens

Copy `secrets.json.template` to `secrets.json`

Replace the value of the key using the above links



# run notebooks

Run `jupyter notebook` and launch a notebook file

Run the imports and do any pip installs you might need



