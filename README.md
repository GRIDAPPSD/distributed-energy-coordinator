# Distributed Energy Coordinator Application

This repository contains the distributed energy coordinator application for over-voltage mitigation in power distribution systems. The application solves a fairness-based energy coordination problem that enables customers to share the responsibility of voltage regulation in distribution systems. Since the focus of this application is on solving the over-voltage problem, it is effective only on the test feeder with high penetration of distributed energy resources (DERs) such as photovoltaics (PVs). For the demonstration purpose, a modified IEEE 123-bus test feeder is created by replacing the nominal loads with secondary configuration and populating the system with customer-owned PVs and utility-scale PVs to generate a hypothetical over-voltage scenario. The CIM XML file of the test case is available inside the inputs/feeder directory of this repository. The focus of this README is not on the application itself, but rather on how to configure/run it.


## Prerequisites

The proposed application is developed in Python, and it requires the following packages for formulating and solving the optimization problem. Note that all these packages can be installed using pip

### GridAPPSD

I am running GridAPPSD through docker so I am following the Windows 10 WSL guide.

[GridAPPS-D Documentation](https://gridappsd-training.readthedocs.io/en/develop/#)

I am not sure if it matters, but I deviated from the install an cloned gridappsd-docker into windows 10. This seems to install cleaner for me.

```shell
git clone git@github.com:GRIDAPPSD/gridappsd-docker.git
cd gridappsd-docker
./stop.sh
./run.sh -t v2023.01.0
```

Once the docker containers are installed make sure they are running and attach to the gridappsd container.

```shell
./run-gridappsd.sh
```

## Setup

```shell
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Config
set the environ variables in auth.py to the point to simulation and message bus configs

```python
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['OUTPUTS'] = f"{ROOT}/outputs"
os.environ['BUS_CONFIG'] = f"{ROOT}/config/system_message_bus.yml"
os.environ['GOSS_CONFIG'] = f"{ROOT}/config/pnnl.goss.gridappsd.cfg"
os.environ['SIM_CONFIG'] = f"{ROOT}/config/ieee123.json"
```

make sure the gridappsd container is running the matching feeder, there is an assert to compare the config files. If there is a missmatch update the goss config and copy it into the docker container and rerun. 

```bash
docker cp config/pnnl.goss.gridappsd.cfg  gridappsd:/gridappsd/conf/pnnl.goss.gridappsd.cfg
```

## Run
run main.py from the termanal in the main directory.

```bash
python main.py
```
