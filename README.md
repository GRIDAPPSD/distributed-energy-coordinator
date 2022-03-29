# Distributed Energy Coordinator Application

This repository contains the distributed energy coordinator application for over-voltage mitigation in power distribution systems. The application is a fairness-based energy coordination strategy enabling customers to share the responsibility of voltage regulation in distribution systems. Since, the focus of this application is on solving the over-voltage problem, it is supposed to work only on the test feeder with high penetration of distributed energy resources (DERs) such as photovoltaics (PVs). For the demonstration purpose, a modified IEEE 123-bus test feeder is created by replacing the nominal loads with secondary configuration and populating system with customer-owned PVs and utility scale PVs to generate hypothetical over-voltage scenario. The CIM XML file of the test case is available inside inputs/feeder directory of this repository. The focus of this README is not on the application itself, but rather how to configure/run it as a user.


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
   
3. Once the platform is up and running, upload the modified IEEE 123-bus feeder into the database. This feeder will be publicly avaliable in future release.

    ```` console
    user@user> cd distributed-energy-coordinator
    user@user/distributed-energy-coordinator> cd inputs/feeder
    user@user> ./go.sh
    
    # These commands will upload the CIM XML of the modified IEEE 123-bus in the database.
    ```` 
   
4. Once the database is updated, start the application using the following commands. Note that the application can be invoked with any feeders in the database. However, this feeder has higher number of PVs populated in the secongary model and hence is most suitable to demonstrate the effectiveness of the proposed approach. 

    ```` console
    user@user> cd distributed-energy-coordinator
    user@user/distributed-energy-coordinator> cd dec
    user@user/distributed-energy-coordinator/dec> python3 run_dec_both.py "feeder_mrid"
    
    # This will start the optimization, and after the convergence, voltages and curtailment factors will be plotted.
    ```` 