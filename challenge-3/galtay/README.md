# Introduction

This code can be used to analyze [high-throughput screening](https://en.wikipedia.org/wiki/High-throughput_screening)
data from https://www.synapse.org/#!Synapse:syn5522627.


# Getting Started 

Create and activate a python environment with your favorite tool. 
An example for conda would be, 

```bash
conda create --name ng310 python=3.10
conda activate ng310
```

Run the following command to install requirements. 

```bash
pip install -r requirements.txt
```


# Setup Data Config

## config.ini 

Copy `config.ini.template` to `config.ini` and edit the line that starts with `DATA_PATH`. 
This should point to an empty directory.
This location will be used to store synapse datasets and derived data.
 

## secrets.json

In order to securely download data from synapse you will need a personal access token. 
Generate one by follwing the instructions at the links below, 

* https://help.synapse.org/docs/Client-Configuration.1985446156.html
* https://www.synapse.org/#!PersonalAccessTokens

Next, copy `secrets.json.template` to the `SECRETS_PATH` specified in the `config.ini` file. 
By default `SECRETS_PATH` = `DATA_PATH/secrets.json` but you can change 
this to whatever you want. Finally, add your personal access token to the `secrets.json` file. 

# Get Synapse Data

Sync the [synapse dataset](https://www.synapse.org/#!Synapse:syn5522627)
to your local machine by running, 

```
python synapse.py
```

You should now see a directory called `syn5522627` in your 
`DATA_PATH/synapse` directory (where `DATA_PATH` was specified in the `config.ini` file)

# Calculate AC50 ratios

Run the following command to generate output files, 

```
python single_agent_screens.py
```
