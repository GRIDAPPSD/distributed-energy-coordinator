# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 10:58:46 2021
@author: Shiva Poudel, Monish Mukherjee
"""

import os
# from area_agent_ADMM_kW import AreaAgent
from area_agent import AreaAgent
# from service_xfmr_agent import Secondary_Agent
# from gridappsd import GridAPPSD, DifferenceBuilder
# from sparql import SPARQLManager
import networkx as nx
import numpy as np
import pandas as pd
import copy
import json
from numpy import linalg as LA
import matplotlib.pyplot as plt

global mult, mult_pv, mult_load, mult_sec_pv

def area_info(G, edge, branch_sw_data, bus_info, sourcebus, area_source_bus):
    # Find area between the switches
    for e in edge:
        G.remove_edge(e[0], e[1])

    # Find area specific information
    # T = list(nx.bfs_tree(G, source = sourcebus).edges())
    # print("\n Number of Buses:", G.number_of_nodes(), "\n", "Number of Edges:", G.number_of_edges())
    # print('\n The number of edges in a Spanning tree is:', len(T))
    # print(list(nx.connected_components(G)))

    # List of sub-graphs. The one that has no sourcebus is the disconnected one
    sp_graph = list(nx.connected_components(G))
    for k in sp_graph:
        if sourcebus == area_source_bus:
            if sourcebus in k:
                area = k
                break
        else:
            if sourcebus not in k:
                area = k
                break

    bus_info_area_i = {}
    idx = 0
    sump = 0
    sumq = 0
    sumpv = 0
    for key, val_bus in bus_info.items():
        if key in area:
            if bus_info[key]['kv'] > 0.4:
                bus_info_area_i[key] = {}
                bus_info_area_i[key]['idx'] = idx
                bus_info_area_i[key]['phases'] = bus_info[key]['phases']
                bus_info_area_i[key]['nodes'] = bus_info[key]['nodes']
                bus_info_area_i[key]['kv'] = bus_info[key]['kv']
                bus_info_area_i[key]['s_rated'] = (bus_info[key]['pv'][0][0] + bus_info[key]['pv'][1][0] + bus_info[key]['pv'][2][0])
                bus_info_area_i[key]['pv'] = [[pv[0] * mult_pv, pv[1] * mult_pv] for pv in bus_info[key]['pv']]
                bus_info_area_i[key]['pq'] = [[pq[0] * mult_load, pq[1] * mult_load] for pq in bus_info[key]['pq']]
                sump += bus_info_area_i[key]['pq'][0][0]
                sump += bus_info_area_i[key]['pq'][1][0]
                sump += bus_info_area_i[key]['pq'][2][0]
                sumq += bus_info_area_i[key]['pq'][0][1]
                sumq += bus_info_area_i[key]['pq'][1][1]
                sumq += bus_info_area_i[key]['pq'][2][1]
                sumpv += bus_info_area_i[key]['pv'][0][0]
                sumpv += bus_info_area_i[key]['pv'][1][0]
                sumpv += bus_info_area_i[key]['pv'][2][0]
                idx += 1

    for key, val_bus in bus_info.items():
        if key in area:
            if bus_info[key]['kv'] < 0.4:
                bus_info_area_i[key] = {}
                bus_info_area_i[key]['idx'] = idx
                bus_info_area_i[key]['phases'] = bus_info[key]['phases']
                bus_info_area_i[key]['nodes'] = bus_info[key]['nodes']
                bus_info_area_i[key]['kv'] = bus_info[key]['kv']
                # bus_info_area_i[key]['pv'] = bus_info[key]['pv']
                bus_info_area_i[key]['pv'] = [i * mult_sec_pv for i in bus_info[key]['pv']]
                bus_info_area_i[key]['pq'] = [i * mult_load for i in bus_info[key]['pq']]
                bus_info_area_i[key]['s_rated'] = (bus_info[key]['pv'][0])
                sump += bus_info_area_i[key]['pq'][0]
                sumq += bus_info_area_i[key]['pq'][1]
                sumpv += bus_info_area_i[key]['pv'][0]
                idx += 1
    idx = 0
    print(sump, sumq, sumpv)

    secondary_model = ['SPLIT_PHASE', 'TPX_LINE']
    branch_sw_data_area_i = {}
    nor_open = ['sw7', 'sw8']
    for key, val_bus in branch_sw_data.items():
        if val_bus['fr_bus'] in bus_info_area_i and val_bus['to_bus'] in bus_info_area_i:
            if branch_sw_data[key]['type'] not in secondary_model and key not in nor_open:
                branch_sw_data_area_i[key] = {}
                branch_sw_data_area_i[key]['idx'] = idx
                branch_sw_data_area_i[key]['type'] = branch_sw_data[key]['type']
                branch_sw_data_area_i[key]['from'] = bus_info_area_i[branch_sw_data[key]['fr_bus']]['idx']
                branch_sw_data_area_i[key]['to'] = bus_info_area_i[branch_sw_data[key]['to_bus']]['idx']
                branch_sw_data_area_i[key]['fr_bus'] = branch_sw_data[key]['fr_bus']
                branch_sw_data_area_i[key]['to_bus'] = branch_sw_data[key]['to_bus']
                branch_sw_data_area_i[key]['phases'] = branch_sw_data[key]['phases']
                if branch_sw_data[key]['type'] == 'SPLIT_PHASE':
                    branch_sw_data_area_i[key]['impedance'] = branch_sw_data[key]['impedance']
                    branch_sw_data_area_i[key]['impedance1'] = branch_sw_data[key]['impedance1']
                else:
                    branch_sw_data_area_i[key]['zprim'] = branch_sw_data[key]['zprim']
                idx += 1
    idx = 0
    for key, val_bus in branch_sw_data.items():
        if val_bus['fr_bus'] in bus_info_area_i and val_bus['to_bus'] in bus_info_area_i:
            if branch_sw_data[key]['type'] in secondary_model:
                branch_sw_data_area_i[key] = {}
                branch_sw_data_area_i[key]['idx'] = idx
                branch_sw_data_area_i[key]['type'] = branch_sw_data[key]['type']
                branch_sw_data_area_i[key]['from'] = bus_info_area_i[branch_sw_data[key]['fr_bus']]['idx']
                branch_sw_data_area_i[key]['to'] = bus_info_area_i[branch_sw_data[key]['to_bus']]['idx']
                branch_sw_data_area_i[key]['fr_bus'] = branch_sw_data[key]['fr_bus']
                branch_sw_data_area_i[key]['to_bus'] = branch_sw_data[key]['to_bus']
                branch_sw_data_area_i[key]['phases'] = branch_sw_data[key]['phases']
                if branch_sw_data[key]['type'] == 'SPLIT_PHASE':
                    branch_sw_data_area_i[key]['impedance'] = branch_sw_data[key]['impedance']
                    branch_sw_data_area_i[key]['impedance1'] = branch_sw_data[key]['impedance1']
                else:
                    branch_sw_data_area_i[key]['impedance'] = branch_sw_data[key]['zprim']
                idx += 1
    return branch_sw_data_area_i, bus_info_area_i


def _main():
    
    f = open('../inputs/feeder/bus_info_ieee123_pri.json')
    bus_info = json.load(f)
    f = open('../inputs/feeder/branch_sw_xfmr_ieee123_pri.json')
    branch_sw_xfmr = json.load(f)
    G = nx.Graph()
    for b in branch_sw_xfmr:
        G.add_edge(branch_sw_xfmr[b]['fr_bus'], branch_sw_xfmr[b]['to_bus'])

    # Finding the switch delimited areas and give the area specific information to agents
    sourcebus = '150'
    area_info_swt = {'area_cen': {}, 'area_1': {}, 'area_2': {}, 'area_3': {}, 'area_4': {}, 'area_5': {}}

    area_info_swt['area_1']['edges'] = [['13', '152'], ['18', '135']]
    area_info_swt['area_1']['source_bus'] = '150'
    edge = area_info_swt['area_1']['edges']
    area_source_bus = area_info_swt['area_1']['source_bus']
    G_area = copy.deepcopy(G)
    branch_sw_data_area_1, bus_info_area_1 = area_info(G_area, edge, branch_sw_xfmr, bus_info, sourcebus,
                                                       area_source_bus)

    area_info_swt['area_2']['edges'] = [['18', '135'], ['151', '300']]
    area_info_swt['area_2']['source_bus'] = '135'
    edge = area_info_swt['area_2']['edges']
    area_source_bus = area_info_swt['area_2']['source_bus']
    G_area = copy.deepcopy(G)
    branch_sw_data_area_2, bus_info_area_2 = area_info(G_area, edge, branch_sw_xfmr, bus_info, sourcebus,
                                                       area_source_bus)

    area_info_swt['area_3']['edges'] = [['60', '160'], ['13', '152'], ['54', '94']]
    area_info_swt['area_3']['source_bus'] = '152'
    edge = area_info_swt['area_3']['edges']
    area_source_bus = area_info_swt['area_3']['source_bus']
    G_area = copy.deepcopy(G)
    branch_sw_data_area_3, bus_info_area_3 = area_info(G_area, edge, branch_sw_xfmr, bus_info, sourcebus,
                                                       area_source_bus)

    area_info_swt['area_4']['edges'] = [['60', '160'], ['97', '197'], ['54', '94']]
    area_info_swt['area_4']['source_bus'] = '160'
    edge = area_info_swt['area_4']['edges']
    area_source_bus = area_info_swt['area_4']['source_bus']
    G_area = copy.deepcopy(G)
    branch_sw_data_area_4, bus_info_area_4 = area_info(G_area, edge, branch_sw_xfmr, bus_info, sourcebus,
                                                       area_source_bus)

    area_info_swt['area_5']['edges'] = [['97', '197'], ['151', '300']]
    area_info_swt['area_5']['source_bus'] = '197'
    edge = area_info_swt['area_5']['edges']
    area_source_bus = area_info_swt['area_5']['source_bus']
    G_area = copy.deepcopy(G)
    branch_sw_data_area_5, bus_info_area_5 = area_info(G_area, edge, branch_sw_xfmr, bus_info, sourcebus,
                                                       area_source_bus)



    bus_info_store = {}
    bus_info_store['area_1'] = bus_info_area_1
    bus_info_store['area_2'] = bus_info_area_2
    bus_info_store['area_3'] = bus_info_area_3
    bus_info_store['area_4'] = bus_info_area_4
    bus_info_store['area_5'] = bus_info_area_5
    
    # json_bi = open('bus_info_store.json', 'w')
    # json.dump(bus_info_store, json_bi, indent=4)
    # json_bi.close()

    v_source = [1.0475, 1.0475, 1.0475]

    # Run the centralized power flow to get the real-time operating voltage
    print("\n\nCheck for Over Voltage: Extracting the network Voltage and Injection")
    area_info_swt['area_cen']['edges'] = [['54', '94'], ['151', '300']]  # [['13', '152'], ['18', '135']]
    area_info_swt['area_cen']['source_bus'] = '150'
    area_info_swt['area_cen']['vsrc'] = v_source
    edge = area_info_swt['area_cen']['edges']
    area_source_bus = area_info_swt['area_cen']['source_bus']
    G_area = copy.deepcopy(G)
    branch_sw_data_area_cen, bus_info_area_cen = area_info(G_area, edge, branch_sw_xfmr, bus_info, sourcebus,
                                                           area_source_bus)
    agent_bus = area_info_swt['area_cen']['source_bus']
    agent_bus_idx = bus_info_area_cen[agent_bus]['idx']
    vsrc = area_info_swt['area_cen']['vsrc']
    area_i_agent = AreaAgent()
    pf_flag = 0
    dist_flag = False
    bus_voltage_area_cen, flow_area_cen, alpha_cen, trash_Qi, trash_Pi = area_i_agent.alpha_area(branch_sw_data_area_cen, bus_info_area_cen,
                                                                             agent_bus, agent_bus_idx, vsrc, pf_flag, dist_flag)
    print("\nCentral optimization, Alpha:", alpha_cen)   

    
    ################ Distributed Optimization ################   
    bus_voltage = {'97': {'A': [], 'B': [], 'C': []}, 
                   '60': {'A': [], 'B': [], 'C': []}, 
                   '13': {'A': [], 'B': [], 'C': []}, 
                   '18': {'A': [], 'B': [], 'C': []}}
       
    for key in bus_voltage:
        bus_voltage[key]['A'] = [bus_voltage_area_cen[key]['A']]
        bus_voltage[key]['B'] = [bus_voltage_area_cen[key]['B']]
        bus_voltage[key]['C'] = [bus_voltage_area_cen[key]['C']] 
   
    bus_voltage_area_4 = {'97': {'A': 1.0475, 'B': 1.0475, 'C': 1.0475}}
    bus_voltage_area_3 = {'60': {'A': 1.0475, 'B': 1.0475, 'C': 1.0475}}
    bus_voltage_area_1 = {'13': {'A': 1.0475, 'B': 1.0475, 'C': 1.0475},
                          '18': {'A': 1.0475, 'B': 1.0475, 'C': 1.0475}}


    # ************************************ Message exchange starts from here ****************************
    # ************************* Take the voltage and injections from respective source node *************
    pf_flag = 0
    dist_flag = True
    alpha_store = {'cen': alpha_cen, 'area_5': [], 'area_4': [], 'area_3': [], 'area_2': [], 'area_1': []}
    
     # mu = 1000
     # mu_v_alpha = [10000, 1]
     # mu_v_alpha = [5000, 5000, 10]
     # mu_v_alpha = [1000, 500, 1]
     # mu_v_alpha =   [1000, 2000, 0]
    
    # mu_v_alpha = [15000, 100000, 100000, 10] ### runs for fixed active power
    mu_v_alpha = [7000, 1000000, 1000000, 5] ### runs for fixed active power
   
    lamdav = np.zeros((5,3))
    lamda =  [0, 0, 0, 0, 0]
    lamda1 = copy.deepcopy(lamda)
    lamda2 = copy.deepcopy(lamda)
    lamda3 = copy.deepcopy(lamda)
    lamda4 = copy.deepcopy(lamda)
    lamda5 = copy.deepcopy(lamda)

    lastVA = [1.0475,  1.0475,  1.0475,  1.0475, 1.0475]
    lastVB = [1.0475,  1.0475,  1.0475,  1.0475, 1.0475]
    lastVC = [1.0475,  1.0475,  1.0475,  1.0475, 1.0475]
    
    alphas = [0.25, 0.5, 0.25, 0.35, 0.65]
    # alphas = [0, 0, 0, 0, 0]    
   
    # lastVA = [1.04, bus_voltage['18']['A'][-1], bus_voltage['13']['A'][-1], bus_voltage['60']['A'][-1], bus_voltage['97']['A'][-1]]
    # lastVB = [1.04, bus_voltage['18']['B'][-1], bus_voltage['13']['B'][-1], bus_voltage['60']['B'][-1], bus_voltage['97']['B'][-1]]
    # lastVC = [1.04, bus_voltage['18']['C'][-1], bus_voltage['13']['C'][-1], bus_voltage['60']['C'][-1], bus_voltage['97']['C'][-1]]
    
    # lastVA_actual = copy.deepcopy(lastVA) 
    # lastVB_actual = copy.deepcopy(lastVB) 
    # lastVC_actual = copy.deepcopy(lastVA) 
   
    All_child_bus = ['13', '18', '60', '97']
   
    lamdaPi = {}
    lamdaQi = {}

    lastPi = {}    
    lastQi = {}

    lastPi_actual= {}
    lastQi_actual= {}
   
    for bus in All_child_bus: 
        lamdaPi[bus] = [0, 0 ,0]
        lamdaQi[bus] = [0, 0, 0]
       
        lastPi[bus]  = [0, 0, 0]
        lastQi[bus]  = [0, 0, 0]

        lastPi_actual[bus] = [0, 0, 0]
        lastQi_actual[bus] = [0, 0, 0]

    baseS = 1 / (1000000 * 100 / 3)

    for k in range(40):
        print("\nIteration:", k + 1)
        pf_flag = 0

        ### Update voltages ###
        bus_voltage['97']['A'].append((bus_voltage_area_4['97']['A']))
        bus_voltage['97']['B'].append((bus_voltage_area_4['97']['B']))
        bus_voltage['97']['C'].append((bus_voltage_area_4['97']['C']))

        bus_voltage['18']['A'].append((bus_voltage_area_1['18']['A']))
        bus_voltage['18']['B'].append((bus_voltage_area_1['18']['B']))
        bus_voltage['18']['C'].append((bus_voltage_area_1['18']['C']))

        bus_voltage['13']['A'].append((bus_voltage_area_1['13']['A']))
        bus_voltage['13']['B'].append((bus_voltage_area_1['13']['B']))
        bus_voltage['13']['C'].append((bus_voltage_area_1['13']['C']))
       
        bus_voltage['60']['A'].append((bus_voltage_area_3['60']['A']))
        bus_voltage['60']['B'].append((bus_voltage_area_3['60']['B']))
        bus_voltage['60']['C'].append((bus_voltage_area_3['60']['C']))

        lastVA_actual = [1.0475, bus_voltage['18']['A'][-1], bus_voltage['13']['A'][-1], bus_voltage['60']['A'][-1], bus_voltage['97']['A'][-1]]
        lastVB_actual = [1.0475, bus_voltage['18']['B'][-1], bus_voltage['13']['B'][-1], bus_voltage['60']['B'][-1], bus_voltage['97']['B'][-1]]
        lastVC_actual = [1.0475, bus_voltage['18']['C'][-1], bus_voltage['13']['C'][-1], bus_voltage['60']['C'][-1], bus_voltage['97']['C'][-1]]

        if k > 1: 
            r_Vi = []; 
            for m in range(5):
                lamdav[m][0] +=  mu_v_alpha[0] * 1 * (lastVA[m]**2 - lastVA_actual[m]**2)
                lamdav[m][1] +=  mu_v_alpha[0] * 1 * (lastVB[m]**2 - lastVB_actual[m]**2)
                lamdav[m][2] +=  mu_v_alpha[0] * 1 * (lastVC[m]**2 - lastVC_actual[m]**2)
                 
                r_Vi += [(lastVA[m]**2 - lastVA_actual[m]**2), (lastVB[m]**2 - lastVB_actual[m]**2), (lastVC[m]**2 - lastVC_actual[m]**2)]
   
            for bus in All_child_bus: 
                lamdaPi[bus][0] +=  mu_v_alpha[1] * (lastPi[bus][0] - lastPi_actual[bus][0]) * baseS
                lamdaPi[bus][1] +=  mu_v_alpha[1] * (lastPi[bus][1] - lastPi_actual[bus][1]) * baseS
                lamdaPi[bus][2] +=  mu_v_alpha[1] * (lastPi[bus][2] - lastPi_actual[bus][2]) * baseS
                 
            for bus in All_child_bus: 
                lamdaQi[bus][0] +=  mu_v_alpha[2] * (lastQi[bus][0] - lastQi_actual[bus][0]) * baseS
                lamdaQi[bus][1] +=  mu_v_alpha[2] * (lastQi[bus][1] - lastQi_actual[bus][1]) * baseS
                lamdaQi[bus][2] +=  mu_v_alpha[2] * (lastQi[bus][2] - lastQi_actual[bus][2]) * baseS
                 
            r_alpha = []
            for m in range(5):
                avg_alpha = sum(alphas)/len(alphas)
                lamda1[m] += mu_v_alpha[3] * (alphas[0] - alphas[m])
                lamda2[m] += mu_v_alpha[3] * (alphas[1] - alphas[m])
                lamda3[m] += mu_v_alpha[3] * (alphas[2] - alphas[m])
                lamda4[m] += mu_v_alpha[3] * (alphas[3] - alphas[m])
                lamda5[m] += mu_v_alpha[3] * (alphas[4] - alphas[m])
           
                r_alpha += [(alphas[0] - alphas[m]), (alphas[1] - alphas[m]), (alphas[2] - alphas[m]),  (alphas[3] - alphas[m]) ,(alphas[4] - alphas[m])]
               

        # AREA 5
        area_no = 5
        child5 = []
        lastVi = [lastVA[area_no-1], lastVB[area_no-1], lastVC[area_no-1]]
        agent_bus = area_info_swt['area_5']['source_bus']
        agent_bus_idx = bus_info_area_5[agent_bus]['idx']
       
        area_info_swt['area_5']['vsrc'] = [bus_voltage['97']['A'][-1], bus_voltage['97']['B'][-1], bus_voltage['97']['C'][-1]]

        vsrc = area_info_swt['area_5']['vsrc']
       
        bus_voltage_area_5, flow_area_5, alpha_5, lastPi, lastQi = area_i_agent.alpha_area(branch_sw_data_area_5, bus_info_area_5, agent_bus,
                                                                                    agent_bus_idx, vsrc, pf_flag, dist_flag,
                                                                                    5, alphas, lamdav[4,:], lamda5, lamdaPi, lamdaQi,
                                                                                    mu_v_alpha, child5, lastPi, lastQi, lastVi)
        v_src_area_5 = bus_voltage_area_5[agent_bus]
        print("\nArea 5 optimization, Alpha:", alpha_5)
       
        lastPi_actual['97'][0] = flow_area_5['l118']['A'][0]
        lastPi_actual['97'][1] = flow_area_5['l118']['B'][0]
        lastPi_actual['97'][2] = flow_area_5['l118']['C'][0]
       
        lastQi_actual['97'][0] = flow_area_5['l118']['A'][1]
        lastQi_actual['97'][1] = flow_area_5['l118']['B'][1]
        lastQi_actual['97'][2] = flow_area_5['l118']['C'][1]

        # AREA 2
        area_no = 2
        child2 = []
        lastVi = [lastVA[area_no-1], lastVB[area_no-1], lastVC[area_no-1]]
        agent_bus = area_info_swt['area_2']['source_bus']
        agent_bus_idx = bus_info_area_2[agent_bus]['idx']
       
        area_info_swt['area_2']['vsrc'] = [bus_voltage['18']['A'][-1], bus_voltage['18']['B'][-1], bus_voltage['18']['C'][-1]]
        vsrc = area_info_swt['area_2']['vsrc']
        bus_voltage_area_2, flow_area_2, alpha_2, lastPi, lastQi = area_i_agent.alpha_area(branch_sw_data_area_2, bus_info_area_2, agent_bus,
                                                                                    agent_bus_idx, vsrc, pf_flag, dist_flag,
                                                                                    2, alphas, lamdav[1,:], lamda2, lamdaPi, lamdaQi,
                                                                                    mu_v_alpha, child2, lastPi, lastQi, lastVi)
        v_src_area_2 = bus_voltage_area_2[agent_bus]
        print("Area 2 optimization, Alpha:", alpha_2)
       
        lastPi_actual['18'][0] = flow_area_2['l114']['A'][0]
        lastPi_actual['18'][1] = flow_area_2['l114']['B'][0]
        lastPi_actual['18'][2] = flow_area_2['l114']['C'][0]
       
        lastQi_actual['18'][0] = flow_area_2['l114']['A'][1]
        lastQi_actual['18'][1] = flow_area_2['l114']['B'][1]
        lastQi_actual['18'][2] = flow_area_2['l114']['C'][1]
       
        # AREA 4
        area_no = 4
        child4 = ['97']
        lastVi = [lastVA[area_no-1], lastVB[area_no-1], lastVC[area_no-1]]
       
        # Update injections from Area 5 before solving Area 4
        # bus_info_area_4['97']['pq'] = [flow_area_5['l118']['A'], flow_area_5['l118']['B'], flow_area_5['l118']['C']]
       
        agent_bus = area_info_swt['area_4']['source_bus']
        agent_bus_idx = bus_info_area_4[agent_bus]['idx']
       
        area_info_swt['area_4']['vsrc'] = [bus_voltage['60']['A'][-1], bus_voltage['60']['B'][-1], bus_voltage['60']['C'][-1]]
        vsrc = area_info_swt['area_4']['vsrc']
       
        bus_voltage_area_4, flow_area_4, alpha_4, lastPi, lastQi = area_i_agent.alpha_area(branch_sw_data_area_4, bus_info_area_4, agent_bus,
                                                                                    agent_bus_idx, vsrc, pf_flag, dist_flag,
                                                                                    4, alphas, lamdav[3,:], lamda4, lamdaPi, lamdaQi,
                                                                                    mu_v_alpha, child4, lastPi, lastQi, lastVi)
        v_src_area_4 = bus_voltage_area_4[agent_bus] 
        print("Area 4 optimization, Alpha:", alpha_4)
       
        lastPi_actual['60'][0] = flow_area_4['l117']['A'][0]
        lastPi_actual['60'][1] = flow_area_4['l117']['B'][0]
        lastPi_actual['60'][2] = flow_area_4['l117']['C'][0]
       
        lastQi_actual['60'][0] = flow_area_4['l117']['A'][1]
        lastQi_actual['60'][1] = flow_area_4['l117']['B'][1]
        lastQi_actual['60'][2] = flow_area_4['l117']['C'][1]
       

        # AREA 3
        area_no = 3
        child3 = ['60']
        lastVi = [lastVA[area_no-1], lastVB[area_no-1], lastVC[area_no-1]]
       
        # Update injections from Area 4 before solving Area 3
        # bus_info_area_3['60']['pq'] = [flow_area_4['l117']['A'], flow_area_4['l117']['B'], flow_area_4['l117']['C']]
       
        agent_bus = area_info_swt['area_3']['source_bus']
        agent_bus_idx = bus_info_area_3[agent_bus]['idx']

        area_info_swt['area_3']['vsrc'] = [bus_voltage['13']['A'][-1], bus_voltage['13']['B'][-1], bus_voltage['13']['C'][-1]]
        vsrc = area_info_swt['area_3']['vsrc']
        bus_voltage_area_3, flow_area_3, alpha_3, lastPi, lastQi = area_i_agent.alpha_area(branch_sw_data_area_3, bus_info_area_3, agent_bus,
                                                                                    agent_bus_idx, vsrc, pf_flag, dist_flag,
                                                                                    3, alphas, lamdav[2,:], lamda3, lamdaPi, lamdaQi,
                                                                                    mu_v_alpha, child3, lastPi, lastQi, lastVi)
        v_src_area_3 = bus_voltage_area_3[agent_bus] 
        print("Area 3 optimization, Alpha:", alpha_3)
       
        lastPi_actual['13'][0] = flow_area_3['l116']['A'][0]
        lastPi_actual['13'][1] = flow_area_3['l116']['B'][0]
        lastPi_actual['13'][2] = flow_area_3['l116']['C'][0]
       
       
        lastQi_actual['13'][0] = flow_area_3['l116']['A'][1]
        lastQi_actual['13'][1] = flow_area_3['l116']['B'][1]
        lastQi_actual['13'][2] = flow_area_3['l116']['C'][1]

        # AREA 1
        area_no = 1
        child1 = ['18', '13']
        lastVi = [lastVA[area_no-1], lastVB[area_no-1], lastVC[area_no-1]]
       
        # Update injections from Area 2 and Area 3 before solving Area 1
        # bus_info_area_1['18']['pq'] = [flow_area_2['l114']['A'], flow_area_2['l114']['B'], flow_area_2['l114']['C']]
        # bus_info_area_1['13']['pq'] = [flow_area_3['l116']['A'], flow_area_3['l116']['B'], flow_area_3['l116']['C']]
       
        # print(bus_info_area_1['18']['pq'], bus_info_area_1['13']['pq'])
        agent_bus = area_info_swt['area_1']['source_bus']
        agent_bus_idx = bus_info_area_1[agent_bus]['idx']
        area_info_swt['area_1']['vsrc'] = v_source
        vsrc = area_info_swt['area_1']['vsrc']
        bus_voltage_area_1, flow_area_1, alpha_1, lastPi, lastQi = area_i_agent.alpha_area(branch_sw_data_area_1, bus_info_area_1, agent_bus,
                                                                                    agent_bus_idx, vsrc, pf_flag, dist_flag,
                                                                                    1, alphas, lamdav[0,:], lamda1, lamdaPi, lamdaQi,
                                                                                    mu_v_alpha, child1, lastPi, lastQi, lastVi)
        v_src_area_1 = bus_voltage_area_1[agent_bus] 
        print("Area 1 optimization, Alpha:", alpha_1)
       

        #### Update injections from Areas ####
        bus_info_area_1['18']['pq'] = [flow_area_2['l114']['A'], flow_area_2['l114']['B'], flow_area_2['l114']['C']]
        bus_info_area_1['13']['pq'] = [flow_area_3['l116']['A'], flow_area_3['l116']['B'], flow_area_3['l116']['C']]
        bus_info_area_3['60']['pq'] = [flow_area_4['l117']['A'], flow_area_4['l117']['B'], flow_area_4['l117']['C']]
        bus_info_area_4['97']['pq'] = [flow_area_5['l118']['A'], flow_area_5['l118']['B'], flow_area_5['l118']['C']]
       
        lastVA = [v_src_area_1['A'], v_src_area_2['A'], v_src_area_3['A'], v_src_area_4['A'], v_src_area_5['A']]
        lastVB = [v_src_area_1['B'], v_src_area_2['B'], v_src_area_3['B'], v_src_area_4['B'], v_src_area_5['B']]
        lastVC = [v_src_area_1['C'], v_src_area_2['C'], v_src_area_3['C'], v_src_area_4['C'], v_src_area_5['C']]
       
        # Store alphas
        alpha_store['area_5'].append(alpha_5)
        alpha_store['area_4'].append(alpha_4)
        alpha_store['area_3'].append(alpha_3)
        alpha_store['area_2'].append(alpha_2)
        alpha_store['area_1'].append(alpha_1)

        s_alpha = [(alphas[0] -alpha_1), (alphas[1] - alpha_2),  (alphas[2] - alpha_3), (alphas[3] - alpha_4), (alphas[4] - alpha_5)]
        alphas = [alpha_1, alpha_2, alpha_3, alpha_4, alpha_5]     
       
       
        s_ViA = [(lastVA[0]**2 - v_src_area_1['A']**2), (lastVA[1]**2 - v_src_area_2['A']**2), (lastVA[2]**2 - v_src_area_3['A']), (lastVA[3]**2 - v_src_area_4['A']**2), (lastVA[4]**2 - v_src_area_5['A']**2)]
        s_ViB = [(lastVB[0]**2 - v_src_area_1['B']**2), (lastVB[1]**2 - v_src_area_2['B']**2), (lastVB[2]**2 - v_src_area_3['B']), (lastVB[3]**2 - v_src_area_4['B']**2), (lastVB[4]**2 - v_src_area_5['B']**2)]
        s_ViC = [(lastVC[0]**2 - v_src_area_1['C']**2), (lastVC[1]**2 - v_src_area_2['C']**2), (lastVC[2]**2 - v_src_area_3['C']), (lastVC[3]**2 - v_src_area_4['C']**2), (lastVC[4]**2 - v_src_area_5['C']**2)]
        s_Vi = s_ViA + s_ViB + s_ViC
               
       
    # for m in range(5):
    #   print('Area{} vdiff-phaseA: {}'.format(m, (lastVA[m] - lastVA_actual[m])))
    #   print('Area{} vdiff-phaseB: {}'.format(m, (lastVB[m] - lastVB_actual[m])))
    #   print('Area{} vdiff-phaseC: {}'.format(m, (lastVC[m] - lastVC_actual[m])))
     
    # for bus in All_child_bus: 
    #    print("Area{} PA-diff Cal vs UpStream Actual {}".format(bus, lastPi[bus][0] -  lastPi_actual[bus][0]))
    #    print("Area{} PB-diff Cal vs UpStream Actual {}".format(bus, lastPi[bus][1] -  lastPi_actual[bus][1]))
    #    print("Area{} PC-diff Cal vs UpStream Actual {}".format(bus, lastPi[bus][2] -  lastPi_actual[bus][2]))
      
    #    print("Area{} QA-diff Cal vs UpStream Actual {}".format(bus, lastQi[bus][0] -  lastQi_actual[bus][0]))
    #    print("Area{} QB-diff Cal vs UpStream Actual {}".format(bus, lastQi[bus][1] -  lastQi_actual[bus][1]))
    #    print("Area{} QC-diff Cal vs UpStream Actual {}".format(bus, lastQi[bus][2] -  lastQi_actual[bus][2]))
             
    plt.plot(alpha_store['area_5'], 'r')
    plt.plot(alpha_store['area_4'], 'k')
    plt.plot(alpha_store['area_3'], 'b')
    plt.plot(alpha_store['area_2'], 'g')
    plt.plot(alpha_store['area_1'], 'c')
    plt.plot(np.ones(k) * alpha_cen, 'r.')

    json_fp = open('../outputs/alpha_store_admm.json', 'w')
    json.dump(alpha_store, json_fp, indent=4)
    json_fp.close()
   
    plt.grid()
    plt.xlabel('Iterations count')
    plt.ylabel('alpha')
    plt.legend(['Area5', 'Area4', 'Area3', 'Area2', 'Area1', 'Centralized'])
    plt.show()
   
    plt.plot(bus_voltage['18']['A'][1:], '-r')
    plt.plot(bus_voltage['18']['B'][1:], '--r')
    plt.plot(bus_voltage['18']['C'][1:], '-.r')
    plt.plot(bus_voltage['97']['A'][1:], '-b')
    plt.plot(bus_voltage['97']['B'][1:], '--b')
    plt.plot(bus_voltage['97']['C'][1:], '-.b')
   
    plt.scatter(k, bus_voltage_area_cen['18']['A'], marker ="o", c="red", alpha=0.2) 
    plt.scatter(k, bus_voltage_area_cen['18']['B'], marker ="^", c="red", alpha=0.2) 
    plt.scatter(k, bus_voltage_area_cen['18']['C'], marker ="*", c="red", alpha=0.2) 
    plt.scatter(k, bus_voltage_area_cen['97']['A'], marker ="o", c="blue", alpha=0.2) 
    plt.scatter(k, bus_voltage_area_cen['97']['B'], marker ="^", c="blue", alpha=0.2) 
    plt.scatter(k, bus_voltage_area_cen['97']['C'], marker ="*", c="blue", alpha=0.2) 
   
    plt.grid()
    plt.xlabel('Iterations count')
    plt.ylabel('Voltage (p.u.)')
    plt.legend(['18A','18B','18C','97A','97B','97C'], loc='upper left')
    plt.show()
    return bus_voltage, alpha_store, alphas, alpha_cen


if __name__ == '__main__':
    global mult, mult_pv, mult_load
    load_mult = pd.read_csv('../inputs/loadshape_15min.csv', header=None)
    pv_mult = pd.read_csv('../inputs/pvshape_15min.csv', header=None)
    alphas = {}
    alpha_store = {}
    bus_voltages = {}
    t = 0    
    for t in range(48, 69):
        mult_pv = pv_mult.iloc[:, 0][t]
        mult_load = load_mult.iloc[:, 0][t]
        mult_sec_pv = mult_pv 
        bus_voltage_t, alpha_store_t, alpha_area_t, alpha_cen_t = _main()        
        bus_voltages[t] = bus_voltage_t
        alphas[t] = {}
        alphas[t]['areas']= alpha_area_t
        alphas[t]['cen']= alpha_cen_t
        alpha_store[t] = alpha_store_t
        print("Running time: ", t)
        exit()