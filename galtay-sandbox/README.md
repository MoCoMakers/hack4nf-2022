# hack4nf
2022 Hack4NF https://hack4nf.bemyapp.com/

# access to GENIE 

Start by requesting access to GENIE dataset - https://www.synapse.org/#!Synapse:syn34548529/wiki/618904


After you get a confirmation email - you will need to go back the the site, click "Request Access" and click Accept on the electronic terms that pop up. You should then have the ability to download from the website (and thus the code).

# downloading from synapse

Synapse.org -> Login (must have account, see hackathon site) -> Generate a code per:

https://help.synapse.org/docs/Client-Configuration.1985446156.html

https://www.synapse.org/#!PersonalAccessTokens

Copy `secrets.json.template` to `secrets.json`

Replace the value of the key using the above links

# install hack4nf python package 

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

# run notebooks

Run `jupyter notebook` and launch a notebook file

Run the imports and do any pip installs you might need



