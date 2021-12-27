# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 10:58:46 2021

@author: poud579
"""


from area_agent import AreaCoordinator
from service_xfmr_agent import Secondary_Agent
from gridappsd import GridAPPSD
from sparql import SPARQLManager
import networkx as nx
import numpy as np
import json
from operator import add


def secondary_info(G, sourcebus, bus_info, tpx_xfmr):
    sp_graph = list(nx.connected_components(G))
    for k in sp_graph:
        if sourcebus in k:
            area = k
            break
    bus_info_sec_agent_i = {}
    idx = 0
    for key, val_bus in bus_info.items():
        if key in area:
            bus_info_sec_agent_i[key] = {}
            bus_info_sec_agent_i[key]['idx'] = idx
            bus_info_sec_agent_i[key]['phase'] = bus_info[key]['phases']
            bus_info_sec_agent_i[key]['nodes'] = bus_info[key]['nodes']
            bus_info_sec_agent_i[key]['injection'] = bus_info[key]['injection']
            idx += 1
    
    idx = 0
    tpx_xfmr_agent_i = {}
    for key, val_bus in tpx_xfmr.items():
        if val_bus['fr_bus'] in bus_info_sec_agent_i and val_bus['to_bus'] in bus_info_sec_agent_i:
            tpx_xfmr_agent_i[key] = {}
            tpx_xfmr_agent_i[key]['idx'] = idx
            tpx_xfmr_agent_i[key]['type']= tpx_xfmr[key]['type']
            tpx_xfmr_agent_i[key] ['from'] = bus_info_sec_agent_i[tpx_xfmr[key]['fr_bus']]['idx']
            tpx_xfmr_agent_i[key] ['to'] =  bus_info_sec_agent_i[tpx_xfmr[key]['to_bus']]['idx']
            tpx_xfmr_agent_i[key] ['fr_bus'] = tpx_xfmr[key]['fr_bus']
            tpx_xfmr_agent_i[key] ['to_bus'] =  tpx_xfmr[key]['to_bus']
            if tpx_xfmr[key]['type'] == 'SPLIT_PHASE':
                tpx_xfmr_agent_i[key] ['impedance'] =  [0.006+0.0136j]
                tpx_xfmr_agent_i[key] ['impedance1'] = [0.012+0.0068j]
            else:
                tpx_xfmr_agent_i[key] ['impedance'] =  [0.0042+0.0023j]
            # branch_sw_data_sec_agent_i[key]['zprim'] = branch_sw_data[key]['zprim']
            idx += 1
    # exit()
    
    return bus_info_sec_agent_i, tpx_xfmr_agent_i


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
    
    bus_info_area_i = {}
    idx = 0
    for key, val_bus in bus_info.items():
        if key in area:
            bus_info_area_i[key] = {}
            bus_info_area_i[key]['idx'] = idx
            bus_info_area_i[key]['phases'] = bus_info[key]['phases']
            bus_info_area_i[key]['nodes'] = bus_info[key]['nodes']
            bus_info_area_i[key]['injection'] = bus_info[key]['injection']
            idx += 1
    idx = 0
    branch_sw_data_area_i = {}
    for key, val_bus in branch_sw_data.items():
        if val_bus['fr_bus'] in bus_info_area_i and val_bus['to_bus'] in bus_info_area_i:
            branch_sw_data_area_i[key] = {}
            branch_sw_data_area_i[key]['idx'] = idx
            branch_sw_data_area_i[key]['type']= branch_sw_data[key]['type']
            branch_sw_data_area_i[key] ['from'] = bus_info_area_i[branch_sw_data[key]['fr_bus']]['idx']
            branch_sw_data_area_i[key] ['to'] =  bus_info_area_i[branch_sw_data[key]['to_bus']]['idx']
            branch_sw_data_area_i[key] ['fr_bus'] = branch_sw_data[key]['fr_bus']
            branch_sw_data_area_i[key] ['to_bus'] =  branch_sw_data[key]['to_bus']
            branch_sw_data_area_i[key]['phases'] = branch_sw_data[key]['phases']
            branch_sw_data_area_i[key]['zprim'] = branch_sw_data[key]['zprim']
            idx += 1
            
    return branch_sw_data_area_i, bus_info_area_i

class AgentData:

    def __init__(self, ybus, cnv, node_name, energy_consumer, der_pv, lines, pxfmrs, txfmrs, switches):
        self.ybus = ybus
        self.cnv = cnv
        self.energy_consumer = energy_consumer
        self.lines = lines
        self.pxfmrs = pxfmrs
        self.txfmrs = txfmrs
        self.switches = switches
        self.node_name = node_name
        self.der_pv = der_pv
    
    # Extract area specific data
    def area_agent(self):
        ybus = self.ybus 
        cnv = self.cnv 
        energy_consumer =  self.energy_consumer 
        lines = self.lines
        pxfmrs = self.pxfmrs 
        txfmrs = self.txfmrs 
        switches = self.switches
        node_name = self.node_name
        der_pv = self.der_pv
        G = nx.Graph() 

        phaseIdx = {'A': '.1', 'B': '.2', 'C': '.3', 's1\ns2': '.1.2', 's2\ns1': '.1.2'}
        pq_inj = {}
        mult = 0.0
        for obj in energy_consumer:
            p = float(obj['p']['value'])
            q = float(obj['q']['value'])
            if obj['phases']['value'] == '':
                pq_inj[obj['bus']['value'].upper() + '.1' ] = (complex(p, q) / 3) * mult
                pq_inj[obj['bus']['value'].upper() + '.2' ] = (complex(p, q) / 3) * mult
                pq_inj[obj['bus']['value'].upper() + '.3' ] = (complex(p, q) / 3) * mult
            else:
                pq_inj[obj['bus']['value'].upper() + phaseIdx[obj['phases']['value']]] = complex(p, q) * mult
        
        pv_inj = {}
        for obj in der_pv:
            p = float(obj['p']['value'])
            q = float(obj['q']['value'])
            if 's' not in obj['phases']['value']:
                pv_inj[obj['bus']['value'].upper() + phaseIdx[obj['phases']['value']]] = complex(p, q) 

        # Extracting bus information
        print('Extracting bus information')
        bus_info = {}
        idx = 0
        for obj in cnv:
            items = obj.split(',')
            if items[0] == 'Bus':
                continue
            if '0.208' in items[1].strip('"'):
                continue
            bus = items[0].strip('"')
            bus_info[bus] = {}
            bus_info[bus]['idx'] = idx
            bus_info[bus]['kv'] = float(items[1].strip('"'))
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
            pv_a = pv_b = pv_c = complex(0, 0)
            # Find injection and populate the bus_info
            if bus + '.1' in pq_inj:
                pq_a = pq_inj[bus + '.1']
            if bus + '.1' in pv_inj:
                pv_a = pv_inj[bus + '.1']

            if bus + '.2' in pq_inj:
                pq_b = pq_inj[bus + '.2']
            if bus + '.2' in pv_inj:
                pv_b = pv_inj[bus + '.2']

            if bus + '.3' in pq_inj:
                pq_c = pq_inj[bus + '.3']
            if bus + '.3' in pv_inj:
                pv_c = pv_inj[bus + '.3']
            sinj_nodes = [pq_a - pv_a, pq_b - pv_b, pq_c - pv_c]
            bus_info[bus]['injection'] = sinj_nodes
            idx += 1

        # Extracting branch information
        print('Extracting branch information')
        branch_sw_data = {}
        idx_line = 0
        for obj in lines:
            if obj['bus1']['value'].upper() not in bus_info:
                continue
            if obj['bus2']['value'].upper() not in bus_info:
                continue
            line_name = obj['name']['value']
            branch_sw_data[line_name] = {}
            branch_sw_data[line_name]['idx'] = idx_line
            branch_sw_data[line_name]['type']= 'LINE'
            if 's1' in obj['phases']['value']:
                ph = [1, 2]
            else:
                ph = [ord(letter) - 64 for letter in  obj['phases']['value']]
            # print(obj['phases']['value'])
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
            # print(branch_sw_data[line_name])
        
        # Extracting the xmfr information (Power Transformer End)
        print('Extracting Transformer information')
        idx_xmfr = 0
        for idx, obj in enumerate(pxfmrs):
            if idx % 2 == 0:
                if obj['bus']['value'].upper() not in bus_info:
                    continue
                if pxfmrs[idx+1]['bus']['value'].upper() not in bus_info:
                    continue
                xmfr_name = obj['xfmr_name']['value']
                branch_sw_data[xmfr_name] = {}
                branch_sw_data[xmfr_name]['idx'] = idx_line + idx_xmfr + 0
                branch_sw_data[xmfr_name]['type']= 'XFMR'
                ph = [ord(letter) - 64 for letter in  'ABC']
                branch_sw_data[xmfr_name]['phases'] = ph
                branch_sw_data[xmfr_name] ['from'] = bus_info[obj['bus']['value'].upper()]['idx']
                branch_sw_data[xmfr_name] ['to'] = bus_info[pxfmrs[idx+1]['bus']['value'].upper()]['idx']
                branch_sw_data[xmfr_name] ['fr_bus'] = obj['bus']['value'].upper()
                branch_sw_data[xmfr_name] ['to_bus'] = pxfmrs[idx+1]['bus']['value'].upper()
                z_prim = np.zeros((3, 3), dtype=complex)
                branch_sw_data[xmfr_name]['zprim'] = z_prim
                idx_xmfr += 1
                G.add_edge(obj['bus']['value'].upper(), pxfmrs[idx+1]['bus']['value'].upper())

        # Extracting the xmfr information (Transformer tank End)
        # ratedU = {}
        # for obj in xmfrs:
        #     xfmr_name = obj['xfmr_name']['value']
        #     enum = int(obj['enum']['value'])
        #     if xfmr_name not in ratedU:
        #         ratedU[xfmr_name] = {}
        #     ratedU[xfmr_name][enum] = int(obj['baseV']['value'])

        for idx, obj in enumerate(txfmrs):
            xmfr_name = obj['xfmr_name']['value']
            if xmfr_name not in branch_sw_data:
                if obj['bus']['value'].upper() not in bus_info:
                    continue
                if txfmrs[idx+1]['bus']['value'].upper() not in bus_info:
                    continue
                branch_sw_data[xmfr_name] = {}
                branch_sw_data[xmfr_name]['idx'] = idx_line + idx_xmfr
                branch_sw_data[xmfr_name]['type']= 'REG'
                ph = [ord(letter) - 64 for letter in  'ABC']
                branch_sw_data[xmfr_name]['phases'] = ph
                branch_sw_data[xmfr_name] ['from'] = bus_info[obj['bus']['value'].upper()]['idx']
                branch_sw_data[xmfr_name] ['to'] = bus_info[txfmrs[idx+1]['bus']['value'].upper()]['idx']
                branch_sw_data[xmfr_name] ['fr_bus'] = obj['bus']['value'].upper()
                branch_sw_data[xmfr_name] ['to_bus'] = txfmrs[idx+1]['bus']['value'].upper()
                z_prim = np.zeros((3, 3), dtype=complex)
                branch_sw_data[xmfr_name]['zprim'] = z_prim
                idx_xmfr += 1
                G.add_edge(obj['bus']['value'].upper(), txfmrs[idx+1]['bus']['value'].upper())
            # print(branch_sw_data[xmfr_name])

        # Extracting the switch information
        print("Extracting switch information")
        sw_pos_base = []
        swt_idxs_ph = []
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
                ph = [1, 2, 3]
                z_prim = np.zeros((len(ph), len(ph)), dtype=complex)
                if obj['isopen']['value'] == 'false':
                    sw_pos_base.append(1)
                    # z_prim = -1 * np.linalg.inv(ybus[np.ix_(fr_node, t_node)])
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
            # print(obj['bus1']['value'].upper(), obj['bus2']['value'].upper())
            G.add_edge(obj['bus1']['value'].upper(), obj['bus2']['value'].upper())
        
        return bus_info, branch_sw_data, G
        
    # Extract secondary specific data
    def secondary_agent(self):
        ybus = self.ybus 
        cnv = self.cnv 
        energy_consumer =  self.energy_consumer 
        lines = self.lines
        pxfmrs = self.pxfmrs 
        txfmrs = self.txfmrs 
        switches = self.switches
        node_name = self.node_name
        der_pv = self.der_pv

        G = nx.Graph() 
        phaseIdx = {'A': '.1', 'B': '.2', 'C': '.3', 's1\ns2': '.1.2', 's2\ns1': '.1.2'}
        pq_inj = {}
        mult = 0.5
        for obj in energy_consumer:
            p = float(obj['p']['value'])
            q = float(obj['q']['value'])
            if obj['phases']['value'] == '':
                pq_inj[obj['bus']['value'].upper() + '.1' ] = (complex(p, q) / 3) * mult
                pq_inj[obj['bus']['value'].upper() + '.2' ] = (complex(p, q) / 3) * mult
                pq_inj[obj['bus']['value'].upper() + '.3' ] = (complex(p, q) / 3) * mult
            else:
                pq_inj[obj['bus']['value'].upper() + phaseIdx[obj['phases']['value']]] = complex(p, q) * mult
        pv_inj = {}
        for obj in der_pv:
            p = float(obj['p']['value'])
            q = float(obj['q']['value'])
            if 's' in obj['phases']['value']:
                pv_inj[obj['bus']['value'].upper() + phaseIdx[obj['phases']['value']]] = complex(p, q) 

        Bus = {}
        service_xfmr_bus = {}
        for obj in txfmrs:
            xfmr_name = obj['xfmr_name']['value']
            enum = int(obj['end_number']['value'])
            # phase = obj['phase']['value']
            if xfmr_name not in Bus:
                Bus[xfmr_name] = {}
            Bus[xfmr_name][enum] = obj['bus']['value']
            if enum == 1:
                phase = obj['phase']['value']
            if enum == 3 and Bus[xfmr_name][1].upper() not in service_xfmr_bus:
                service_xfmr_bus[Bus[xfmr_name][1].upper()] = {}
                service_xfmr_bus[Bus[xfmr_name][1].upper()]['phase'] = phase
        # Extracting bus information
        s = 0
        print('Extracting bus information')
        bus_info = {}
        idx = 0
        for obj in cnv:
            items = obj.split(',')
            bus = items[0].strip('"')
            if items[0] == 'Bus':
                continue
            if '0.208' in items[1].strip('"') or bus in service_xfmr_bus:
                bus_info[bus] = {}
                bus_info[bus]['idx'] = idx
                bus_info[bus]['kv'] = float(items[1].strip('"'))
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
                sinj_nodes = np.zeros((1), dtype=complex)
                pq_s = complex(0, 0)
                pv_s = complex(0, 0)
                # Find injection and populate the bus_info
                if bus + '.1.2' in pq_inj:
                    pq_s = pq_inj[bus + '.1.2']
                    s += pq_s.imag
                    pv_s = pv_inj[bus + '.1.2']
                sinj_nodes = [pq_s - pv_s]
                bus_info[bus]['injection'] = sinj_nodes
                idx += 1

        # Extracting branch information
        print('Extracting branch information')
        tpx_xfmr = {}
        idx_line = 0
        for obj in lines:
            bus1, bus2 = obj['bus1']['value'].upper(), obj['bus2']['value'].upper()
            if bus1 in bus_info and bus2 in bus_info:
                if bus_info[bus1]['kv'] == 0.208:
                    line_name = obj['name']['value']
                    tpx_xfmr[line_name] = {}
                    tpx_xfmr[line_name]['idx'] = idx_line
                    tpx_xfmr[line_name]['type']= 'LINE'
                    ph = [1, 2]
                    tpx_xfmr[line_name] ['from'] = bus_info[obj['bus1']['value'].upper()]['idx']
                    tpx_xfmr[line_name] ['to'] = bus_info[obj['bus2']['value'].upper()]['idx']
                    tpx_xfmr[line_name] ['fr_bus'] = bus1
                    tpx_xfmr[line_name] ['to_bus'] = bus2
                    fr_node = []
                    t_node = []
                    for p in ph:
                        from_node = obj['bus1']['value'].upper() + '.' + str(p)
                        to_node = obj['bus2']['value'].upper() + '.' + str(p)
                        fr_node.append(node_name[from_node]) 
                        t_node.append(node_name[to_node]) 
                    z = -1*np.linalg.inv(ybus[np.ix_(fr_node, t_node)])
                    z_prim = np.zeros((2, 2), dtype=complex)
                    # Making the matrix 2 x 2 for triplex line
                    for p in range(len(ph)):
                        for q in range(len(ph)):
                            z_prim[ph[p]-1][ph[q]-1] = z[p][q]
                    tpx_xfmr[line_name]['impedance'] = [0.0042+0.0023j]
                    idx_line += 1
                    G.add_edge(obj['bus1']['value'].upper(), obj['bus2']['value'].upper())

        # print(len(tpx_xfmr))
        # print(len(bus_info))
        # Extracting the xmfr information (Power Transformer End)
        # SORRY- PowerTransformerEnd will not be a part of secondary agent

        # Extracting the xmfr information (Transformer tank End)
        # ratedU = {}
        # for obj in xmfrs:
        #     xfmr_name = obj['xfmr_name']['value']
        #     enum = int(obj['enum']['value'])
        #     if xfmr_name not in ratedU:
        #         ratedU[xfmr_name] = {}
        #     ratedU[xfmr_name][enum] = int(obj['baseV']['value'])
        idx_xmfr = 0
        for idx, obj in enumerate(txfmrs):
            xmfr_name = obj['xfmr_name']['value']
            if xmfr_name not in tpx_xfmr:
                if obj['bus']['value'].upper() in bus_info and txfmrs[idx+1]['bus']['value'].upper() in bus_info:
                    tpx_xfmr[xmfr_name] = {}
                    tpx_xfmr[xmfr_name]['idx'] = idx_line + idx_xmfr
                    tpx_xfmr[xmfr_name]['type']= 'SPLIT_PHASE'
                    tpx_xfmr[xmfr_name] ['from'] = bus_info[obj['bus']['value'].upper()]['idx']
                    tpx_xfmr[xmfr_name] ['to'] = bus_info[txfmrs[idx+1]['bus']['value'].upper()]['idx']
                    tpx_xfmr[xmfr_name] ['fr_bus'] = obj['bus']['value'].upper()
                    tpx_xfmr[xmfr_name] ['to_bus'] = txfmrs[idx+1]['bus']['value'].upper()
                    tpx_xfmr[xmfr_name] ['impedance'] =  [0.006+0.0136j]
                    tpx_xfmr[xmfr_name] ['impedance1'] = [0.012+0.0068j]
                    idx_xmfr += 1
                    G.add_edge(obj['bus']['value'].upper(), txfmrs[idx+1]['bus']['value'].upper())

            # print(branch_sw_data[xmfr_name])
        return bus_info, tpx_xfmr, G, service_xfmr_bus


def _main():
    
    gapps = GridAPPSD()
    simulation_id = '725830594'
    feeder_mrid = "_C1C3E687-6FFD-C753-582B-632A27E28507"
    # feeder_mrid = "_E407CBB6-8C8D-9BC9-589C-AB83FBF0826D"
    feeder_mrid = "_59AD8E07-3BF9-A4E2-CB8F-C3722F837B62"
    model_api_topic = "goss.gridappsd.process.request.data.powergridmodel"

    # Query Grid Data
    print('Querying grid data \n')
    sparql_mgr = sparql_mgr = SPARQLManager(gapps, feeder_mrid, model_api_topic, simulation_id)
    ysparse, nodelist = sparql_mgr.ybus_export()
    energy_consumer = sparql_mgr.query_energyconsumer_lf()
    der_pv = sparql_mgr.query_photovoltaic()
    lines = sparql_mgr.PerLengthPhaseImpedance_line_names()
    pxfmrs = sparql_mgr.PowerTransformerEnd_xfmr_names()
    txfmrs = sparql_mgr.TransformerTank_xfmr_names()
    switches = sparql_mgr.SwitchingEquipment_switch_names()

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
    
    print('Extracting agents data')
    agent_input = AgentData(ybus, cnv, node_name, energy_consumer, der_pv, lines, pxfmrs, txfmrs, switches)

    ########################### SECONDARY AGENTS #################################
    # Extracting secondary agents data from the grid data
    bus_info, tpx_xfmr, G, service_xfmr_bus = agent_input.secondary_agent()

    # Extract the inputs required for each secondary agents
    # Store the equivalent injection to pass into area agent
    message_injection = {}
    for agent_bus in service_xfmr_bus:
        # Extracting a single agent data from secondary agent data
        bus_info_sec_agent_i, tpx_xfmr_agent_i = secondary_info(G, agent_bus, bus_info, tpx_xfmr)

        # Invoke optimization with secondary agent location and indices
        sec_i_agent = Secondary_Agent()
        agent_bus_idx = bus_info_sec_agent_i[agent_bus]['idx']
        vsrc = [1.05]
        sec_inj = sec_i_agent.alpha_area(tpx_xfmr_agent_i, bus_info_sec_agent_i, agent_bus, agent_bus_idx, vsrc, service_xfmr_bus)  
        message_injection[agent_bus] = sec_inj

    ########################### COORDINATING AGENT ###############################
    # Extracting area agent on 4.16 kv level from the grid data
    bus_info, branch_sw_data, G = agent_input.area_agent()
    for k in message_injection:
        bus_info[k]['injection'] = list(map(add, message_injection[k], bus_info[k]['injection']))

    # Finding the switch delimited areas and give the area specific information to agents    
    # edge = [['18', '135'], ['151', '300_OPEN']]
    # edge = [['60', '160'], ['97', '197'], ['54', '94_OPEN']]
    # edge = [['151', '300'], ['97', '197']]
    # sourcebus = '150'
    # branch_sw_data_area_i, bus_info_area_i = area_info(G, edge, branch_sw_data, bus_info, sourcebus)

    print('\n Invoke area agent')
    area_i_agent = AreaCoordinator()
    agent_bus = '150R'
    agent_bus_idx = bus_info[agent_bus]['idx']
    vsrc = [1.0, 1.0, 1.0]
    area_i_agent.alpha_area(branch_sw_data,  bus_info, agent_bus, agent_bus_idx, vsrc)


if __name__ == '__main__':
    _main()
