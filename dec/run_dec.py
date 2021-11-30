# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 10:58:46 2021

@author: poud579
"""


from area_agent import AreaCoordinator
from gridappsd import GridAPPSD
from sparql import SPARQLManager
import networkx as nx


def _main():
    
    print('Querying Ybus')
    gapps = GridAPPSD()
    simulation_id = '725830594'
    feeder_mrid = "_C1C3E687-6FFD-C753-582B-632A27E28507"
    model_api_topic = "goss.gridappsd.process.request.data.powergridmodel"
    sparql_mgr = sparql_mgr = SPARQLManager(gapps, feeder_mrid, model_api_topic, simulation_id)
    ysparse, nodelist = sparql_mgr.ybus_export()
    
    # Node list into dictionary
    node_name = {}
    for idx, obj in enumerate(nodelist):
        node_name[obj.strip('\"')] = idx
    cnv = sparql_mgr.vnom_export()
    
    # Find an area and give the area specific information to agents    
    area1_agent = AreaCoordinator()
    area1_agent.alpha()

if __name__ == '__main__':
    _main()
