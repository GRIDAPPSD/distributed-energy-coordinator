# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 10:58:46 2021
@author: poud579
"""

import os
from area_agent_ekw import alpha_area
import networkx as nx
import numpy as np
import pandas as pd
import copy
import json
import matplotlib.pyplot as plt
from numpy import linalg as LA
from power_flow import AreaAlpha
import timeit


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
                bus_info_area_i[key]['s_rated'] = (
                            bus_info[key]['pv'][0][0] + bus_info[key]['pv'][1][0] + bus_info[key]['pv'][2][0])
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


def split_graph(bus_info, branch_sw_xfmr):
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

    v_source = [1.0475, 1.0475, 1.0475]

    # Run the centralized power flow to get the real-time operating voltage
    area_info_swt['area_cen']['edges'] = [['54', '94'], ['151', '300']]  # [['13', '152'], ['18', '135']]
    area_info_swt['area_cen']['source_bus'] = '150'
    area_info_swt['area_cen']['vsrc'] = v_source
    edge = area_info_swt['area_cen']['edges']
    area_source_bus = area_info_swt['area_cen']['source_bus']
    G_area = copy.deepcopy(G)
    branch_sw_data_area_cen, bus_info_area_cen = area_info(G_area, edge, branch_sw_xfmr, bus_info, sourcebus,
                                                           area_source_bus)

    areas_info = {'bus_info': {}, 'branch_info': {}}
    areas_info['bus_info']['area_1'] = bus_info_area_1
    areas_info['bus_info']['area_2'] = bus_info_area_2
    areas_info['bus_info']['area_3'] = bus_info_area_3
    areas_info['bus_info']['area_4'] = bus_info_area_4
    areas_info['bus_info']['area_5'] = bus_info_area_5
    areas_info['bus_info']['area_cenarea_cen'] = bus_info_area_cen

    areas_info['branch_info']['area_1'] = branch_sw_data_area_1
    areas_info['branch_info']['area_2'] = branch_sw_data_area_2
    areas_info['branch_info']['area_3'] = branch_sw_data_area_3
    areas_info['branch_info']['area_4'] = branch_sw_data_area_4
    areas_info['branch_info']['area_5'] = branch_sw_data_area_5
    areas_info['branch_info']['area_cen'] = branch_sw_data_area_cen

    return areas_info, area_info_swt


def dist_linear_pf(area_info, area_info_swt, bus_voltage, alpha_bisection):
    bus_info_area_1 = area_info['bus_info']['area_1']
    branch_sw_data_area_1 = area_info['branch_info']['area_1']
    bus_info_area_2 = area_info['bus_info']['area_2']
    branch_sw_data_area_2 = area_info['branch_info']['area_2']
    bus_info_area_3 = area_info['bus_info']['area_3']
    branch_sw_data_area_3 = area_info['branch_info']['area_3']
    bus_info_area_4 = area_info['bus_info']['area_4']
    branch_sw_data_area_4 = area_info['branch_info']['area_4']
    bus_info_area_5 = area_info['bus_info']['area_5']
    branch_sw_data_area_5 = area_info['branch_info']['area_5']

    # We are doing parallel so need 6 rounds for convergence.
    vmax = 1
    count = 0
    while vmax > 0.0001:
    # for k in range(7):
        pf_flag = 1
        dist_flag = True

        # AREA 5
        agent_bus = area_info_swt['area_5']['source_bus']
        agent_bus_idx = bus_info_area_5[agent_bus]['idx']

        area_info_swt['area_5']['vsrc'] = [bus_voltage['97']['A'][-1], bus_voltage['97']['B'][-1],
                                           bus_voltage['97']['C'][-1]]

        vsrc = area_info_swt['area_5']['vsrc']

        start = timeit.timeit()

        bus_voltage_area_5, flow_area_5, alpha_5 = alpha_area(branch_sw_data_area_5, bus_info_area_5,
                                                                           agent_bus,
                                                                           agent_bus_idx, vsrc, pf_flag, dist_flag,
                                                                           alpha_bisection[4])
        # print("Area 5 optimization, Alpha:", alpha_5)
        end = timeit.timeit()
        # print('Time', start - end)

        # AREA 2
        agent_bus = area_info_swt['area_2']['source_bus']
        agent_bus_idx = bus_info_area_2[agent_bus]['idx']

        area_info_swt['area_2']['vsrc'] = [bus_voltage['18']['A'][-1], bus_voltage['18']['B'][-1],
                                           bus_voltage['18']['C'][-1]]
        vsrc = area_info_swt['area_2']['vsrc']
        bus_voltage_area_2, flow_area_2, alpha_2 = alpha_area(branch_sw_data_area_2, bus_info_area_2,
                                                                           agent_bus,
                                                                           agent_bus_idx, vsrc, pf_flag, dist_flag,
                                                                           alpha_bisection[1])
        # print("Area 2 optimization, Alpha:", alpha_2)

        # AREA 4
        # Update injections from Area 5 before solving Area 4
        # bus_info_area_4['97']['pq'] = [flow_area_5['l118']['A'], flow_area_5['l118']['B'], flow_area_5['l118']['C']]

        agent_bus = area_info_swt['area_4']['source_bus']
        agent_bus_idx = bus_info_area_4[agent_bus]['idx']

        area_info_swt['area_4']['vsrc'] = [bus_voltage['60']['A'][-1], bus_voltage['60']['B'][-1],
                                           bus_voltage['60']['C'][-1]]
        vsrc = area_info_swt['area_4']['vsrc']

        bus_voltage_area_4, flow_area_4, alpha_4 = alpha_area(branch_sw_data_area_4, bus_info_area_4,
                                                                           agent_bus,
                                                                           agent_bus_idx, vsrc, pf_flag, dist_flag,
                                                                           alpha_bisection[3])
        # print("Area 4 optimization, Alpha:", alpha_4)

        # AREA 3
        # Update injections from Area 4 before solving Area 3
        # bus_info_area_3['60']['pq'] = [flow_area_4['l117']['A'], flow_area_4['l117']['B'], flow_area_4['l117']['C']]

        agent_bus = area_info_swt['area_3']['source_bus']
        agent_bus_idx = bus_info_area_3[agent_bus]['idx']

        area_info_swt['area_3']['vsrc'] = [bus_voltage['13']['A'][-1], bus_voltage['13']['B'][-1],
                                           bus_voltage['13']['C'][-1]]
        vsrc = area_info_swt['area_3']['vsrc']
        bus_voltage_area_3, flow_area_3, alpha_3 = alpha_area(branch_sw_data_area_3, bus_info_area_3,
                                                                           agent_bus,
                                                                           agent_bus_idx, vsrc, pf_flag, dist_flag,
                                                                           alpha_bisection[2])
        # print("Area 3 optimization, Alpha:", alpha_3)

        # AREA 1
        # Update injections from Area 2 and Area 3 before solving Area 1
        # bus_info_area_1['18']['pq'] = [flow_area_2['l114']['A'], flow_area_2['l114']['B'], flow_area_2['l114']['C']]
        # bus_info_area_1['13']['pq'] = [flow_area_3['l116']['A'], flow_area_3['l116']['B'], flow_area_3['l116']['C']]
        agent_bus = area_info_swt['area_1']['source_bus']
        agent_bus_idx = bus_info_area_1[agent_bus]['idx']
        area_info_swt['area_1']['vsrc'] = [1.0475, 1.0475, 1.0475]
        vsrc = area_info_swt['area_1']['vsrc']
        bus_voltage_area_1, flow_area_1, alpha_1 = alpha_area(branch_sw_data_area_1, bus_info_area_1,
                                                                           agent_bus,
                                                                           agent_bus_idx, vsrc, pf_flag, dist_flag,
                                                                           alpha_bisection[0])

        bus_info_area_1['18']['pq'] = [flow_area_2['l114']['A'], flow_area_2['l114']['B'], flow_area_2['l114']['C']]
        bus_info_area_1['13']['pq'] = [flow_area_3['l116']['A'], flow_area_3['l116']['B'], flow_area_3['l116']['C']]
        bus_info_area_3['60']['pq'] = [flow_area_4['l117']['A'], flow_area_4['l117']['B'], flow_area_4['l117']['C']]
        bus_info_area_4['97']['pq'] = [flow_area_5['l118']['A'], flow_area_5['l118']['B'], flow_area_5['l118']['C']]

        # Append voltage and use the latest one for PCC. This appended values could be useful for plotting
        bus_voltage['97']['A'].append(bus_voltage_area_4['97']['A'])
        bus_voltage['97']['B'].append(bus_voltage_area_4['97']['B'])
        bus_voltage['97']['C'].append(bus_voltage_area_4['97']['C'])

        bus_voltage['18']['A'].append(bus_voltage_area_1['18']['A'])
        bus_voltage['18']['B'].append(bus_voltage_area_1['18']['B'])
        bus_voltage['18']['C'].append(bus_voltage_area_1['18']['C'])

        bus_voltage['13']['A'].append(bus_voltage_area_1['13']['A'])
        bus_voltage['13']['B'].append(bus_voltage_area_1['13']['B'])
        bus_voltage['13']['C'].append(bus_voltage_area_1['13']['C'])

        bus_voltage['60']['A'].append(bus_voltage_area_3['60']['A'])
        bus_voltage['60']['B'].append(bus_voltage_area_3['60']['B'])
        bus_voltage['60']['C'].append(bus_voltage_area_3['60']['C'])

        v_diff_5 = max([abs(bus_voltage['97']['A'][-1] - bus_voltage['97']['A'][-2]),
                       abs(bus_voltage['97']['B'][-1] - bus_voltage['97']['B'][-2]),
                       abs(bus_voltage['97']['C'][-1] - bus_voltage['97']['C'][-2])])

        v_diff_4 = max([abs(bus_voltage['60']['A'][-1] - bus_voltage['60']['A'][-2]),
                       abs(bus_voltage['60']['B'][-1] - bus_voltage['60']['B'][-2]),
                       abs(bus_voltage['60']['C'][-1] - bus_voltage['60']['C'][-2])])

        v_diff_3 = max([abs(bus_voltage['13']['A'][-1] - bus_voltage['13']['A'][-2]),
                       abs(bus_voltage['13']['B'][-1] - bus_voltage['13']['B'][-2]),
                       abs(bus_voltage['13']['C'][-1] - bus_voltage['13']['C'][-2])])

        v_diff_2 = max([abs(bus_voltage['18']['A'][-1] - bus_voltage['18']['A'][-2]),
                       abs(bus_voltage['18']['B'][-1] - bus_voltage['18']['B'][-2]),
                       abs(bus_voltage['18']['C'][-1] - bus_voltage['18']['C'][-2])])
        vmax = max([v_diff_5, v_diff_4, v_diff_3, v_diff_2])
        count += 1
    voltage = {}
    for b in bus_voltage_area_1:
        voltage[b] = {}
        voltage[b]['A'] = bus_voltage_area_1[b]['A']
        voltage[b]['B'] = bus_voltage_area_1[b]['B']
        voltage[b]['C'] = bus_voltage_area_1[b]['C']
    for b in bus_voltage_area_2:
        voltage[b] = {}
        voltage[b]['A'] = bus_voltage_area_2[b]['A']
        voltage[b]['B'] = bus_voltage_area_2[b]['B']
        voltage[b]['C'] = bus_voltage_area_2[b]['C']
    for b in bus_voltage_area_3:
        voltage[b] = {}
        voltage[b]['A'] = bus_voltage_area_3[b]['A']
        voltage[b]['B'] = bus_voltage_area_3[b]['B']
        voltage[b]['C'] = bus_voltage_area_3[b]['C']
    for b in bus_voltage_area_4:
        voltage[b] = {}
        voltage[b]['A'] = bus_voltage_area_4[b]['A']
        voltage[b]['B'] = bus_voltage_area_4[b]['B']
        voltage[b]['C'] = bus_voltage_area_4[b]['C']
    for b in bus_voltage_area_5:
        voltage[b] = {}
        voltage[b]['A'] = bus_voltage_area_5[b]['A']
        voltage[b]['B'] = bus_voltage_area_5[b]['B']
        voltage[b]['C'] = bus_voltage_area_5[b]['C']
    return voltage, count


if __name__ == '__main__':
    global mult, mult_pv, mult_load
    plot_flag = True
    # load_mult = pd.read_csv('/home/poud579/git/distributed-energy-coordinator/PESGM_2023/inputs/loadshape_15min.csv',
    #                         header=None)
    # pv_mult = pd.read_csv('/home/poud579/git/distributed-energy-coordinator/PESGM_2023/inputs/PVshape_15min.csv',
    #                       header=None)


    # Load branch and bus info
    f = open('bus_info_ieee123_pri.json')
    bus_info = json.load(f)
    f = open('branch_sw_xfmr_ieee123_pri.json')
    branch_sw_xfmr = json.load(f)

    # Start the time series simulation
    bus_voltages = {}
    for t in range(52, 96):
        # print("Running time...................................... ", t)
        # mult_pv = pv_mult.iloc[:, 0][t]
        # mult_load = load_mult.iloc[:, 0][t]

        mult_load = 0.37
        mult_pv = 0.88
        mult_sec_pv = mult_pv
        print(t, mult_pv, mult_load, mult_sec_pv)

        # Split the network into 5 areas
        areas_info, area_info_swt = split_graph(bus_info, branch_sw_xfmr)


        # Run centralized optimization
        agent_bus = area_info_swt['area_cen']['source_bus']
        bus_info_area_cen = areas_info['bus_info']['area_cenarea_cen']
        branch_sw_data_area_cen = areas_info['branch_info']['area_cen']
        agent_bus_idx = bus_info_area_cen[agent_bus]['idx']
        vsrc = area_info_swt['area_cen']['vsrc']
        # area_i_agent = AreaAgent()

        # ------------------------------------------------
        # # Centralized alpha
        # pf_flag = 0
        # dist_flag = False
        # alpha_bisection = 0.0
        # bus_voltage_area_cen, flow_area_cen, alpha_cen = alpha_area(branch_sw_data_area_cen,
        #                                                                          bus_info_area_cen,
        #                                                                          agent_bus, agent_bus_idx, vsrc,
        #                                                                          pf_flag, dist_flag, alpha_bisection)
        # print("\nCentral optimization, Alpha:", alpha_cen)
        # ------------------------------------------------
        # exit()

        # Centralized power flow; no optimization
        pf_flag = 1
        dist_flag = False
        alpha_bisection = 0.0
        bus_voltage_area_cen, flow_area_cen, alpha_no_use = alpha_area(branch_sw_data_area_cen,
                                                                                    bus_info_area_cen,
                                                                                    agent_bus, agent_bus_idx, vsrc,
                                                                                    pf_flag, dist_flag, alpha_bisection)
        # bus_voltages[t] = bus_voltage_area_cen

        # json_fp = open('bus_voltages_pf.json', 'w')
        # json.dump(bus_voltages, json_fp, indent=4)
        # json_fp.close()
        # area_alpha = AreaAlpha()
        # pf_flag = 0
        # bus_info_area_1 = areas_info['bus_info']['area_1']
        # bus_info_area_2 = areas_info['bus_info']['area_2']
        # bus_info_area_3 = areas_info['bus_info']['area_3']
        # bus_info_area_4 = areas_info['bus_info']['area_4']
        # bus_info_area_5 = areas_info['bus_info']['area_5']

        # bus_voltage_area_1, alpha_cen = area_alpha.area_alpha(branch_sw_data_area_cen, bus_info_area_cen,
        #                                                     agent_bus, agent_bus_idx, vsrc,
        #                                                     bus_info_area_1, bus_info_area_2,
        #                                                     bus_info_area_3, bus_info_area_4,
        #                                                     bus_info_area_5, pf_flag)
        # print("\nCentral optimization, Alpha:", alpha_cen)

        # Extract the initial voltages for PCC and send it to distributed power flow function.
        pcc = ['97', '60', '13', '18']
        bus_voltage = {}
        for bus in pcc:
            bus_voltage[bus] = {}
            bus_voltage[bus]['A'] = [bus_voltage_area_cen[bus]['A']]
            bus_voltage[bus]['B'] = [bus_voltage_area_cen[bus]['B']]
            bus_voltage[bus]['C'] = [bus_voltage_area_cen[bus]['C']]
            # print(bus, bus_voltage_area_cen[bus]['A'], bus_voltage_area_cen[bus]['B'], bus_voltage_area_cen[bus]['C'])

        # Start the bisection method for curtailment
        alpha_range = [0.0, 1]
        alpha_store = []
        volt_store = []
        tol_alpha = 0.001
        tol_voltage = 0.0001
        alpha_voltage = {}
        iter = 0
        total_iter = 0
        while abs(alpha_range[0] - alpha_range[1]) >= tol_alpha:
            alpha_bisection = np.ones(5) * sum(alpha_range) / len(alpha_range)
            alpha_store.append(alpha_bisection[0])
            # print("Bisection:", alpha_range, iter)
            bus_voltage_pf, count = dist_linear_pf(areas_info, area_info_swt, bus_voltage, alpha_bisection)
            # exit()
            # if overvoltage exists, update the bisection
            Va = []
            Vb = []
            Vc = []
            for b in bus_voltage_pf:
                Va.append(bus_voltage_pf[b]['A'])
                Vb.append(bus_voltage_pf[b]['B'])
                Vc.append(bus_voltage_pf[b]['C'])
            V_max = max([max(Va), max(Vb), max(Vc)])
            # print(V_max)
            volt_store.append(V_max)
            alpha_voltage[iter] = {}
            ar = alpha_range
            alpha_voltage[iter]['alpha_range'] = [alpha_range[0], alpha_range[1]]
            alpha_voltage[iter]['maxv'] = V_max
            alpha_voltage[iter]['alpha_bisection'] = alpha_bisection[0]
            alpha_voltage[iter]['iteration'] = count
            total_iter += count
            print(count)
            iter += 1
            if abs(1.05 - V_max) > tol_voltage:
                if 1.05 > V_max:
                    alpha_range[1] = alpha_bisection[0]
                else:
                    alpha_range[0] = alpha_bisection[0]
            else:
                print('I am here', abs(1.05 - V_max))
                break
            # iter += 1
            # if max(Va) > 1.05 or max(Vb) > 1.05 or max(Vc) > 1.05:
            #     alpha_range[0] = alpha_bisection[0]
            #     V = max([max(Va), max(Vb), max(Vc)])
            #     if abs(V - 1.05) < 0.001:
            #         break
            # else:
            #     alpha_range[1] = alpha_bisection[0]

        print(alpha_store, max(Va), max(Vb), max(Vc), total_iter)
        # exit()
        # break

        if plot_flag==True:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.scatter(range(1, len(Va) + 1), Va, c='white', edgecolor='green')
            ax1.scatter(range(1, len(Va) + 1), Vb, c='white', edgecolor='green')
            ax1.scatter(range(1, len(Va) + 1), Vc, c='white', edgecolor='green')

            for idx, v in enumerate(Va):
                if v > 1.05:
                    ax1.scatter(idx, v, c='red', edgecolor='red')
            for idx, v in enumerate(Vb):
                if v > 1.05:
                    ax1.scatter(idx, v, c='red', edgecolor='red')
            for idx, v in enumerate(Vc):
                if v > 1.05:
                    ax1.scatter(idx, v, c='red', edgecolor='red')
            ax1.set_xlabel('Bus Indices', fontsize=18)
            ax1.set_ylabel('Voltage (p.u.)', fontsize=18)
            ax1.plot(range(1, len(Va) + 1), np.ones(len(Vc)) * 1.05, 'r--')
            ax1.set_xlim(1, len(Va) + 1)
            ax1.tick_params(axis='x', labelsize=18)
            ax1.tick_params(axis='y', labelsize=18)

            ax2.plot(range(1, iter + 1), alpha_store, marker='o', color='b')
            # ax2.plot(np.ones(iter + 1) * alpha_cen, 'r', linestyle='--')
            ax2.set_xlim(1, iter)
            ax2.legend(loc="upper right")
            ax2.legend(['Search based', 'Centralized Optimization'], fontsize=16)
            ax2.tick_params(axis='x', labelsize=18)
            ax2.tick_params(axis='y', labelsize=18)

            # plt.grid()
            # plt.xlabel('Iterations count')
            # plt.ylabel('alpha')
            # plt.legend(['Search-based', 'Centralized Optimization'])
            # plt.show()

            ax2.set_xlabel('Itreations', fontsize=18)
            ax2.set_ylabel('Curtailment factor', fontsize=18)
            ax2.tick_params(axis='x')
            ax2.tick_params(axis='y')
            # plt.grid(True)
            plt.show()

            json_fp = open('alpha_store_bisection_low_tol.json', 'w')
            json.dump(alpha_voltage, json_fp, indent=4)
            json_fp.close()

            json_fp = open('bus_voltage_pf_low_tol.json', 'w')
            json.dump(bus_voltage_pf, json_fp, indent=4)
            json_fp.close()
            exit()

    #
    #     # Start the bisection method for different alphas
    #     alpha_range = {'1': [0, 1], '2': [0, 1], '3': [0, 1], '4': [0, 1], '5': [0, 1]}
    #     alpha_bisection = [0, 0, 0, 0, 0]
    #     alpha_store = []
    #     for iter in range(50):
    #         for k in range(5):
    #             alpha_bisection[k] = sum(alpha_range[str(k+1)])/2
    #         bus_voltage_pf = dist_linear_pf(areas_info, area_info_swt, bus_voltage, alpha_bisection)
    #         # if overvoltage exists, update the bisection
    #         V1 = []
    #         V2 = []
    #         V3 = []
    #         V4 = []
    #         V5 = []
    #         for b in bus_voltage_pf:
    #             if b in bus_info_area_1:
    #                 V1.append(bus_voltage_pf[b]['A'])
    #                 V1.append(bus_voltage_pf[b]['B'])
    #                 V1.append(bus_voltage_pf[b]['C'])
    #             if b in bus_info_area_2:
    #                 V2.append(bus_voltage_pf[b]['A'])
    #                 V2.append(bus_voltage_pf[b]['B'])
    #                 V2.append(bus_voltage_pf[b]['C'])
    #             if b in bus_info_area_3:
    #                 V3.append(bus_voltage_pf[b]['A'])
    #                 V3.append(bus_voltage_pf[b]['B'])
    #                 V3.append(bus_voltage_pf[b]['C'])
    #             if b in bus_info_area_4:
    #                 V4.append(bus_voltage_pf[b]['A'])
    #                 V4.append(bus_voltage_pf[b]['B'])
    #                 V4.append(bus_voltage_pf[b]['C'])
    #             if b in bus_info_area_5:
    #                 V5.append(bus_voltage_pf[b]['A'])
    #                 V5.append(bus_voltage_pf[b]['B'])
    #                 V5.append(bus_voltage_pf[b]['C'])
    #
    #         if max(V1) > 1.05:
    #             alpha_range['1'][0] = alpha_bisection[0]
    #         else:
    #             alpha_range['1'][0] = 0
    #             alpha_range['1'][1] = alpha_bisection[0]
    #
    #         if max(V2) > 1.05:
    #             alpha_range['2'][0] = alpha_bisection[1]
    #         else:
    #             alpha_range['2'][0] = 0
    #             alpha_range['2'][1] = alpha_bisection[1]
    #
    #         if max(V3) > 1.05:
    #             alpha_range['3'][0] = alpha_bisection[2]
    #         else:
    #             alpha_range['3'][0] = 0
    #             alpha_range['3'][1] = alpha_bisection[2]
    #
    #         if max(V4) > 1.05:
    #             alpha_range['4'][0] = alpha_bisection[3]
    #         else:
    #             alpha_range['4'][0] = 0
    #             alpha_range['4'][1] = alpha_bisection[3]
    #
    #         if max(V5) > 1.05:
    #             alpha_range['5'][0] = alpha_bisection[4]
    #         else:
    #             alpha_range['5'][0] = 0
    #             alpha_range['5'][1] = alpha_bisection[4]
    #
    #         print("Bisection:", alpha_bisection)
    #
    # # import matplotlib.pyplot as plt
    # # plt.plot(alpha_store, linestyle='--', marker='o', color='g')
    # # plt.plot(np.ones(iter+1) * alpha_cen, 'r')
    #
    # # # plt.plot(store_alpha_2, 'b')
    # # plt.grid()
    # # plt.xlabel('Iterations count')
    # # plt.ylabel('alpha')
    # # plt.legend(['Search-based', 'Centralized Optimization'])
    # # plt.show()
    #     exit()
