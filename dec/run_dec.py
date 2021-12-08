# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 10:58:46 2021

@author: poud579
"""


from area_agent import AreaCoordinator
from gridappsd import GridAPPSD
from sparql import SPARQLManager
import networkx as nx
import numpy as np
import json


def area_info(G, edge, branch_sw_data, bus_info, sourcebus):
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
        if sourcebus not in k:
            area = k
            break
    
    bus_info_area1 = {}
    idx = 0
    for key, val_bus in bus_info.items():
        if key in area:
            bus_info_area1[key] = {}
            bus_info_area1[key]['idx'] = idx
            bus_info_area1[key]['phases'] = bus_info[key]['phases']
            bus_info_area1[key]['nodes'] = bus_info[key]['nodes']
            bus_info_area1[key]['injection'] = bus_info[key]['injection']
            idx += 1
    idx = 0
    branch_sw_data_area1 = {}
    for key, val_bus in branch_sw_data.items():
        if val_bus['fr_bus'] in bus_info_area1 and val_bus['to_bus'] in bus_info_area1:
            branch_sw_data_area1[key] = {}
            branch_sw_data_area1[key]['idx'] = idx
            branch_sw_data_area1[key]['type']= branch_sw_data[key]['type']
            branch_sw_data_area1[key] ['from'] = bus_info_area1[branch_sw_data[key]['fr_bus']]['idx']
            branch_sw_data_area1[key] ['to'] =  bus_info_area1[branch_sw_data[key]['to_bus']]['idx']
            branch_sw_data_area1[key] ['fr_bus'] = branch_sw_data[key]['fr_bus']
            branch_sw_data_area1[key] ['to_bus'] =  branch_sw_data[key]['to_bus']
            branch_sw_data_area1[key]['phases'] = branch_sw_data[key]['phases']
            branch_sw_data_area1[key]['zprim'] = branch_sw_data[key]['zprim']
            idx += 1
            
    return branch_sw_data_area1, bus_info_area1


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

    N = len(node_name)    
    ybus = np.zeros((N, N), dtype=complex)
    for obj in ysparse:
        items = obj.split(',')
        if items[0] == 'Row': # skip header line
            continue
        ybus[int(items[0])-1] [int(items[1])-1] = ybus[int(items[1])-1] [int(items[0])-1] = complex(float(items[2]), float(items[3]))

    G = nx.Graph() 

    print("Querying nodal injections")
    energy_consumer = sparql_mgr.query_energyconsumer_lf()
    phaseIdx = {'A': '.1', 'B': '.2', 'C': '.3', 's1': '.1', 's2': '.2'}
    pq_inj = {}
    mult = 1.0
    for obj in energy_consumer:
        p = float(obj['p']['value'])
        q = float(obj['q']['value'])
        if obj['phases']['value'] == '':
            pq_inj[obj['bus']['value'] + '.1' ] = (complex(p, q) / 3) * mult
            pq_inj[obj['bus']['value'] + '.2' ] = (complex(p, q) / 3) * mult
            pq_inj[obj['bus']['value'] + '.3' ] = (complex(p, q) / 3) * mult
        else:
            pq_inj[obj['bus']['value'] + phaseIdx[obj['phases']['value']]] = complex(p, q) * mult

    print('Querying bus information')
    # Collecting bus and branch information.
    bus_info = {}
    idx = 0
    for obj in cnv:
        items = obj.split(',')
        if items[0] == 'Bus':
            continue

        bus = items[0].strip('"')
        bus_info[bus] = {}
        bus_info[bus]['idx'] = idx
        ph = []
        nodes = []
        node1 = items[2].strip()
        ph.append(node1)
        node = bus + '.' + node1
        nodes.append(node_name[node]) 
        node2 = items[6].strip()
        if node2 != '0':
            ph.append(node2)
            node = bus + '.' + node2
            nodes.append(node_name[node]) 

            node3 = items[10].strip()
            if node3 != '0':
                ph.append(node3)
                node = bus + '.' + node3
                nodes.append(node_name[node]) 

        bus_info[bus]['phases'] = ph
        bus_info[bus]['nodes'] = nodes
        sinj_nodes = np.zeros((3), dtype=complex)
        pq_a = pq_b = pq_c = complex(0, 0)
        # Find injection and populate the bus_info
        if bus + '.1' in pq_inj:
            pq_a = pq_inj[bus + '.1']
        if bus + '.2' in pq_inj:
            pq_b = pq_inj[bus + '.2']
        if bus + '.3' in pq_inj:
            pq_c = pq_inj[bus + '.3']

        sinj_nodes = [pq_a, pq_b, pq_c]
        bus_info[bus]['injection'] = sinj_nodes
        idx += 1

    print('Querying line information')
    lines = sparql_mgr.PerLengthPhaseImpedance_line_names()
    branch_sw_data = {}
    idx_line = 0
    for obj in lines:
        line_name = obj['name']['value']
        branch_sw_data[line_name] = {}
        branch_sw_data[line_name]['idx'] = idx_line
        branch_sw_data[line_name]['type']= 'LINE'
        ph = [ord(letter) - 64 for letter in  obj['phases']['value']]
        branch_sw_data[line_name]['phases'] = ph
        branch_sw_data[line_name] ['from'] = bus_info[obj['bus1']['value'].upper()]['idx']
        branch_sw_data[line_name] ['to'] = bus_info[obj['bus2']['value'].upper()]['idx']
        branch_sw_data[line_name] ['fr_bus'] = obj['bus1']['value'].upper()
        branch_sw_data[line_name] ['to_bus'] = obj['bus2']['value'].upper()
        fr_node = []
        t_node = []
        for p in ph:
            from_node = obj['bus1']['value'].upper() + '.' + str(p)
            to_node = obj['bus2']['value'].upper() + '.' + str(p)
            fr_node.append(node_name[from_node]) 
            t_node.append(node_name[to_node]) 
        z = -1*np.linalg.inv(ybus[np.ix_(fr_node, t_node)])
        z_prim = np.zeros((3, 3), dtype=complex)
        # Making the matrix 3 x 3 for non 3-phase line
        for p in range(len(ph)):
            for q in range(len(ph)):
                z_prim[ph[p]-1][ph[q]-1] = z[p][q]
        # if line_name == 'l115':
        #     print(z_prim)
        #     exit()
        branch_sw_data[line_name]['zprim'] = z_prim
        idx_line += 1
        G.add_edge(obj['bus1']['value'].upper(), obj['bus2']['value'].upper())
    
    # From the xmfr information (Power Transformer End)
    print('Querying xfmr information')
    xmfrs = sparql_mgr.PowerTransformerEnd_xfmr_names()
    idx_xmfr = 0
    for idx, obj in enumerate(xmfrs):
        if idx % 2 == 0:
            xmfr_name = obj['xfmr_name']['value']
            branch_sw_data[xmfr_name] = {}
            branch_sw_data[xmfr_name]['idx'] = idx_line + idx_xmfr + 0
            branch_sw_data[xmfr_name]['type']= 'XFMR'
            ph = [ord(letter) - 64 for letter in  'ABC']
            branch_sw_data[xmfr_name]['phases'] = ph
            branch_sw_data[xmfr_name] ['from'] = bus_info[obj['bus']['value'].upper()]['idx']
            branch_sw_data[xmfr_name] ['to'] = bus_info[xmfrs[idx+1]['bus']['value'].upper()]['idx']
            branch_sw_data[xmfr_name] ['fr_bus'] = obj['bus']['value'].upper()
            branch_sw_data[xmfr_name] ['to_bus'] = xmfrs[idx+1]['bus']['value'].upper()
            z_prim = np.zeros((3, 3), dtype=complex)
            branch_sw_data[xmfr_name]['zprim'] = z_prim
            idx_xmfr += 1
            G.add_edge(obj['bus']['value'].upper(), xmfrs[idx+1]['bus']['value'].upper())

    # From the xmfr information (Transformer Tank End)
    xmfrs = sparql_mgr.TransformerTank_xfmr_names()
    reg_rem = ['reg3a', 'reg4a', 'reg4b']
    for idx, obj in enumerate(xmfrs):
        if idx % 2 == 0 and obj['xfmr_name']['value'] not in reg_rem:
            xmfr_name = obj['xfmr_name']['value']
            branch_sw_data[xmfr_name] = {}
            branch_sw_data[xmfr_name]['idx'] = idx_line + idx_xmfr
            branch_sw_data[xmfr_name]['type']= 'REG'
            ph = [ord(letter) - 64 for letter in  'ABC']
            branch_sw_data[xmfr_name]['phases'] = ph
            branch_sw_data[xmfr_name] ['from'] = bus_info[obj['bus']['value'].upper()]['idx']
            branch_sw_data[xmfr_name] ['to'] = bus_info[xmfrs[idx+1]['bus']['value'].upper()]['idx']
            branch_sw_data[xmfr_name] ['fr_bus'] = obj['bus']['value'].upper()
            branch_sw_data[xmfr_name] ['to_bus'] = xmfrs[idx+1]['bus']['value'].upper()
            z_prim = np.zeros((3, 3), dtype=complex)
            branch_sw_data[xmfr_name]['zprim'] = z_prim
            idx_xmfr += 1
            G.add_edge(obj['bus']['value'].upper(), xmfrs[idx+1]['bus']['value'].upper())
    
    # from the switch information
    print('Querying switch information')
    sw_pos_base = []
    swt_idxs_ph = []
    switches = sparql_mgr.SwitchingEquipment_switch_names()
    switch_info = []
    sw_pos = {'false': 1, 'true': 0}
    idx_swt = 0
    for obj in switches:
        swt_name = obj['name']['value']
        # Storing only the normally closed switch. We only work with radial network for this app.
        if obj['isopen']['value'] == 'false':
            branch_sw_data[swt_name] = {}
            branch_sw_data[swt_name]['idx'] = idx_line + idx_xmfr + idx_swt 
            branch_sw_data[swt_name]['type']= 'SWITCH'
            if obj['phases']['value'] == '':
                obj['phases']['value'] = 'ABC'
            ph = [ord(letter) - 64 for letter in  obj['phases']['value']]
            branch_sw_data[swt_name]['phases'] = ph
            branch_sw_data[swt_name] ['from'] = bus_info[obj['bus1']['value'].upper()]['idx']
            branch_sw_data[swt_name] ['to'] = bus_info[obj['bus2']['value'].upper()]['idx']
            branch_sw_data[swt_name] ['fr_bus'] = obj['bus1']['value'].upper()
            branch_sw_data[swt_name] ['to_bus'] = obj['bus2']['value'].upper()
            fr_node = []
            t_node = []
            swt_idxs = []
            for i, p in enumerate(ph):
                from_node = obj['bus1']['value'].upper() + '.' + str(p)
                to_node = obj['bus2']['value'].upper() + '.' + str(p)
                fr_node.append(node_name[from_node]) 
                t_node.append(node_name[to_node]) 
                swt_idxs.append([fr_node[i], t_node[i]])

            swt_idxs_ph.append(swt_idxs)
            z_prim = np.zeros((len(ph), len(ph)), dtype=complex)
            if obj['isopen']['value'] == 'false':
                sw_pos_base.append(1)
                z_prim = -1 * np.linalg.inv(ybus[np.ix_(fr_node, t_node)])
            else:
                sw_pos_base.append(0)
                nsa = np.array([-1e3+1e3j, -1e3+1e3j, -1e3+1e3j])
                z_prim = -1 * np.linalg.inv(np.diag(-nsa))
            branch_sw_data[swt_name]['zprim'] = z_prim
            message = dict(swt_name = swt_name,   
                            sw_id = idx_swt,                     
                            bus1 = obj['bus1']['value'].upper(),
                            bus2 = obj['bus2']['value'].upper(),
                            phases = obj['phases']['value'],
                            status = sw_pos[obj['isopen']['value']])
            idx_swt += 1
            switch_info.append(message)  
        G.add_edge(obj['bus1']['value'].upper(), obj['bus2']['value'].upper())
    
    # Finding the switch delimited areas
    edge = [['18', '135'], ['151', '300']]
    edge = [['60', '160'], ['97', '197'], ['54', '94']]
    edge = [['151', '300'], ['97', '197']]
    sourcebus = '150'
    branch_sw_data_area1, bus_info_area1 = area_info(G, edge, branch_sw_data, bus_info, sourcebus)
    
    # Give the area specific information to agents    
    area1_agent = AreaCoordinator()
    agent_bus = '197'
    agent_bus_idx = bus_info_area1[agent_bus]['idx']
    vsrc = [0.946, 0.986, 0.959]
    # vsrc = [1.0, 1.0, 1.0]
    area1_agent.alpha_area(branch_sw_data_area1,  bus_info_area1, agent_bus, agent_bus_idx, vsrc)

    # agent_bus = '150R'
    # agent_bus_idx = bus_info[agent_bus]['idx']
    # area1_agent.alpha_area(branch_sw_data,  bus_info, agent_bus, agent_bus_idx)

if __name__ == '__main__':
    _main()