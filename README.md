# Distributed Energy Coordinator Application

This repository contains the distributed energy coordinator application for over-voltage mitigation in power distribution systems. The application is a fairness-based energy coordination strategy enabling customers to share the responsibility of voltage regulation in distribution systems. Since the focus of this application is on solving the over-voltage problem, it is effective only on the test feeder with high penetration of distributed energy resources (DERs) such as photovoltaics (PVs). For the demonstration purpose, a modified IEEE 123-bus test feeder is created by replacing the nominal loads with secondary configuration and populating the system with customer-owned PVs and utility-scale PVs to generate a hypothetical over-voltage scenario. The CIM XML file of the test case is available inside the inputs/feeder directory of this repository. The focus of this README is not on the application itself, but rather on how to configure/run it as a user.


## Prerequisites

The proposed application is developed in Python and it requires the following packages for formulating and solving the optimization problem. Note that all these packages can be installed using pip.

1.  cvxpy
2.  networkx
3.  tabulate

## Running the distributed energy coordinator application

1. From the command line execute the following commands to clone the repository

    ```console
    user@user> git clone https://github.com/GRIDAPPSD/distributed-energy-coordinator
    user@user> cd distributed-energy-coordinator
    ```

2. Run the gridappsd docker. Use the develop tag to download the containers

    ```` console
    user@user> cd gridappsd-docker
    user@user> ./run.sh -t develop
    
    # You will now be inside the container, the following starts gridappsd
    
    gridappsd@f4ede7dacb7d:/gridappsd$ ./run-gridappsd.sh
    ```` 
   
3. Once the platform is up and running, start the application using the following commands. To simulate the over-voltage scenarios in the network, it is assumed that the PV generation is at its peak and nominal loads are scaled by a constant factor; this emulates an operating condition for a particular time of the day. In GridAPPS-D simulation, such operating conditions will be extracted from the simulation output using Simulation API. Once the application is invoked, it will start the optimization, and after the convergence, voltages and curtailment factors will be plotted. A CIM difference message will be created for the PV setpoint, and it will be dumped to a JSON file inside dec/outputs directory. Individual JSON files will be created for each agent that will contain the device information and its corresponding setpoints.

    ```` console
    user@user> cd distributed-energy-coordinator
    user@user/distributed-energy-coordinator> cd dec
    user@user/distributed-energy-coordinator/dec> python3 run_dec_both.py "feeder_mrid"
    ```` 
   
4. A modified IEEE 123-bus test case is also provided with the application. The following commands will upload the modified IEEE 123-bus feeder into the database. This feeder has a higher number of PVs populated in the secondary model and hence is most suitable to demonstrate the effectiveness of the proposed approach. It will be available within the Blazegraph database in the future release.

    ```` console
    user@user> cd distributed-energy-coordinator
    user@user/distributed-energy-coordinator> cd inputs/feeder
    user@user> ./go.sh
    
    # These commands will upload the CIM XML of the modified IEEE 123-bus in the database.
    ````

