# Distributed Energy Coordinator Application

This repository contains the distributed energy coordinator application for over-voltage mitigation in power distribution systems. The application is a fairness-based energy coordination strategy enabling customers to share the responsibility of voltage regulation in distribution systems. Since, the focus of this application is on solving the over-voltage problem, it is supposed to work only on the test feeder with high penetration of distributed energy resources (DERs) such as photovoltaics (PVs). For the demonstration purpose, a modified IEEE 123-bus test feeder is created by replacing the nominal loads with secondary configuration and populating system with customer-owned PVs and utility scale PVs to generate hypothetical over-voltage scenario. The CIM XML file of the test case is available inside inputs/feeder directory of this repository. The focus of this README is not on the application itself, but rather how to configure/run it as a user.


## Prerequisites

The proposed application is developed in Python and it requires the following packages for formulating and solving the optimization problem. Note that all these packages can be installed using pip.

1.  cvxpy
2.  networkx
3.  tabulate
