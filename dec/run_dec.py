# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 10:58:46 2021

@author: poud579
"""
import os
import sys
from area_agent import AreaCoordinator
from service_xfmr_agent import Secondary_Agent
from gridappsd import GridAPPSD, DifferenceBuilder
from sparql import SPARQLManager
import networkx as nx
import numpy as np
from operator import add
import copy
import json
import matplotlib.pyplot as plt


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
            bus_info_area_i[key]['pv'] = bus_info[key]['pv']
            bus_info_area_i[key]['pq'] = bus_info[key]['pq']
            idx += 1
    idx = 0
    branch_sw_data_area_i = {}
    for key, val_bus in branch_sw_data.items():
        if val_bus['fr_bus'] in bus_info_area_i and val_bus['to_bus'] in bus_info_area_i:
            branch_sw_data_area_i[key] = {}
            branch_sw_data_area_i[key]['idx'] = idx
            branch_sw_data_area_i[key]['type'] = branch_sw_data[key]['type']
            branch_sw_data_area_i[key]['from'] = bus_info_area_i[branch_sw_data[key]['fr_bus']]['idx']
            branch_sw_data_area_i[key]['to'] = bus_info_area_i[branch_sw_data[key]['to_bus']]['idx']
            branch_sw_data_area_i[key]['fr_bus'] = branch_sw_data[key]['fr_bus']
            branch_sw_data_area_i[key]['to_bus'] = branch_sw_data[key]['to_bus']
            branch_sw_data_area_i[key]['phases'] = branch_sw_data[key]['phases']
            branch_sw_data_area_i[key]['zprim'] = branch_sw_data[key]['zprim']
            idx += 1
    return branch_sw_data_area_i, bus_info_area_i


class AgentData:

    def __init__(self, ybus, cnv, node_name, energy_consumer, der_pv, lines, pxfmrs, txfmrs, txfmrs_r, txfmrs_z,
                 switches, load_mult, pv_mult):
        self.ybus = ybus
        self.cnv = cnv
        self.energy_consumer = energy_consumer
        self.lines = lines
        self.pxfmrs = pxfmrs
        self.txfmrs = txfmrs
        self.switches = switches
        self.node_name = node_name
        self.der_pv = der_pv
        self.txfmrs_r = txfmrs_r
        self.txfmrs_z = txfmrs_z
        self.load_mult = load_mult
        self.pv_mult = pv_mult

    # Extract area specific data
    def area_agent(self):
        ybus = self.ybus
        cnv = self.cnv
        energy_consumer = self.energy_consumer
        lines = self.lines
        pxfmrs = self.pxfmrs
        txfmrs = self.txfmrs
        switches = self.switches
        node_name = self.node_name
        der_pv = self.der_pv
        G = nx.Graph()

        phaseIdx = {'A': '.1', 'B': '.2', 'C': '.3', 's1\ns2': '.1.2', 's2\ns1': '.1.2'}
        pq_inj = {}
        pv_inj = {}

        for obj in energy_consumer:
            p = float(obj['p']['value'])
            q = float(obj['q']['value'])
            if obj['phases']['value'] == '':
                pq_inj[obj['bus']['value'].upper() + '.1'] = (complex(p, q) / 3) * self.load_mult
                pq_inj[obj['bus']['value'].upper() + '.2'] = (complex(p, q) / 3) * self.load_mult
                pq_inj[obj['bus']['value'].upper() + '.3'] = (complex(p, q) / 3) * self.load_mult
            else:
                pq_inj[obj['bus']['value'].upper() + phaseIdx[obj['phases']['value']]] = complex(p, q) * self.load_mult

        for obj in der_pv:
            p = float(obj['p']['value'])
            q = float(obj['q']['value'])
            if 's' not in obj['phases']['value']:
                if obj['phases']['value'] == '':
                    pv_inj[obj['bus']['value'].upper() + '.1'] = (complex(p, q) / 3) * self.pv_mult
                    pv_inj[obj['bus']['value'].upper() + '.2'] = (complex(p, q) / 3) * self.pv_mult
                    pv_inj[obj['bus']['value'].upper() + '.3'] = (complex(p, q) / 3) * self.pv_mult
                else:
                    pv_inj[obj['bus']['value'].upper() + phaseIdx[obj['phases']['value']]] = complex(p, q) * self.pv_mult

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
            bus_info[bus]['pq'] = [pq_a, pq_b, pq_c]
            bus_info[bus]['pv'] = [pv_a, pv_b, pv_c]
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
            branch_sw_data[line_name]['type'] = 'LINE'
            if obj['phases']['value'] == '':
                obj['phases']['value'] = 'ABC'
            if 's1' in obj['phases']['value']:
                ph = [1, 2]
            else:
                ph = [ord(letter) - 64 for letter in obj['phases']['value'] if letter != 'N']
                if ph[-1] > 3:
                    ph = ph[:-1]
            # print(obj['phases']['value'])
            branch_sw_data[line_name]['phases'] = ph
            branch_sw_data[line_name]['from'] = bus_info[obj['bus1']['value'].upper()]['idx']
            branch_sw_data[line_name]['to'] = bus_info[obj['bus2']['value'].upper()]['idx']
            branch_sw_data[line_name]['fr_bus'] = obj['bus1']['value'].upper()
            branch_sw_data[line_name]['to_bus'] = obj['bus2']['value'].upper()
            fr_node = []
            t_node = []
            for p in ph:
                from_node = obj['bus1']['value'].upper() + '.' + str(p)
                to_node = obj['bus2']['value'].upper() + '.' + str(p)
                fr_node.append(node_name[from_node])
                t_node.append(node_name[to_node])
            z = -1 * np.linalg.inv(ybus[np.ix_(fr_node, t_node)])
            z_prim = np.zeros((3, 3), dtype=complex)
            # Making the matrix 3 x 3 for non 3-phase line
            for p in range(len(ph)):
                for q in range(len(ph)):
                    z_prim[ph[p] - 1][ph[q] - 1] = z[p][q]
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
            xmfr_name = obj['xfmr_name']['value']
            if xmfr_name not in branch_sw_data:
                if obj['bus']['value'].upper() not in bus_info:
                    continue
                if pxfmrs[idx + 1]['bus']['value'].upper() not in bus_info:
                    continue

                branch_sw_data[xmfr_name] = {}
                branch_sw_data[xmfr_name]['idx'] = idx_line + idx_xmfr + 0
                branch_sw_data[xmfr_name]['type'] = 'XFMR'
                ph = [ord(letter) - 64 for letter in 'ABC']
                branch_sw_data[xmfr_name]['phases'] = ph
                branch_sw_data[xmfr_name]['from'] = bus_info[obj['bus']['value'].upper()]['idx']
                branch_sw_data[xmfr_name]['to'] = bus_info[pxfmrs[idx + 1]['bus']['value'].upper()]['idx']
                branch_sw_data[xmfr_name]['fr_bus'] = obj['bus']['value'].upper()
                branch_sw_data[xmfr_name]['to_bus'] = pxfmrs[idx + 1]['bus']['value'].upper()
                z_prim = np.zeros((3, 3), dtype=complex)
                branch_sw_data[xmfr_name]['zprim'] = z_prim
                idx_xmfr += 1
                G.add_edge(obj['bus']['value'].upper(), pxfmrs[idx + 1]['bus']['value'].upper())

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
                if txfmrs[idx + 1]['bus']['value'].upper() not in bus_info:
                    continue
                branch_sw_data[xmfr_name] = {}
                branch_sw_data[xmfr_name]['idx'] = idx_line + idx_xmfr
                branch_sw_data[xmfr_name]['type'] = 'REG'
                ph = [ord(letter) - 64 for letter in 'ABC']
                branch_sw_data[xmfr_name]['phases'] = ph
                branch_sw_data[xmfr_name]['from'] = bus_info[obj['bus']['value'].upper()]['idx']
                branch_sw_data[xmfr_name]['to'] = bus_info[txfmrs[idx + 1]['bus']['value'].upper()]['idx']
                branch_sw_data[xmfr_name]['fr_bus'] = obj['bus']['value'].upper()
                branch_sw_data[xmfr_name]['to_bus'] = txfmrs[idx + 1]['bus']['value'].upper()
                z_prim = np.zeros((3, 3), dtype=complex)
                branch_sw_data[xmfr_name]['zprim'] = z_prim
                idx_xmfr += 1
                G.add_edge(obj['bus']['value'].upper(), txfmrs[idx + 1]['bus']['value'].upper())
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
                branch_sw_data[swt_name]['type'] = 'SWITCH'
                if obj['phases']['value'] == '':
                    obj['phases']['value'] = 'ABC'
                ph = [ord(letter) - 64 for letter in obj['phases']['value']]
                branch_sw_data[swt_name]['phases'] = ph
                branch_sw_data[swt_name]['from'] = bus_info[obj['bus1']['value'].upper()]['idx']
                branch_sw_data[swt_name]['to'] = bus_info[obj['bus2']['value'].upper()]['idx']
                branch_sw_data[swt_name]['fr_bus'] = obj['bus1']['value'].upper()
                branch_sw_data[swt_name]['to_bus'] = obj['bus2']['value'].upper()
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
                    nsa = np.array([-1e3 + 1e3j, -1e3 + 1e3j, -1e3 + 1e3j])
                    z_prim = -1 * np.linalg.inv(np.diag(-nsa))
                branch_sw_data[swt_name]['zprim'] = z_prim
                message = dict(swt_name=swt_name,
                               sw_id=idx_swt,
                               bus1=obj['bus1']['value'].upper(),
                               bus2=obj['bus2']['value'].upper(),
                               phases=obj['phases']['value'],
                               status=sw_pos[obj['isopen']['value']])
                idx_swt += 1
                switch_info.append(message)
                # print(obj['bus1']['value'].upper(), obj['bus2']['value'].upper())
            G.add_edge(obj['bus1']['value'].upper(), obj['bus2']['value'].upper())

        return bus_info, branch_sw_data, G

    # Extract secondary specific data
    def secondary_agent(self):
        ybus = self.ybus
        cnv  = self.cnv
        energy_consumer = self.energy_consumer
        lines  = self.lines
        pxfmrs = self.pxfmrs
        txfmrs = self.txfmrs
        switches  = self.switches
        node_name = self.node_name
        der_pv   = self.der_pv
        txfmrs_r = self.txfmrs_r
        txfmrs_z = self.txfmrs_z


        G = nx.Graph()
        phaseIdx = {'A': '.1', 'B': '.2', 'C': '.3', 's1\ns2': '.1.2', 's2\ns1': '.1.2'}
        pq_inj = {}
        mult = 0.25
        pv_inj = {}
        for obj in energy_consumer:
            p = float(obj['p']['value'])
            q = float(obj['q']['value'])
            if obj['phases']['value'] == '':
                pq_inj[obj['bus']['value'].upper() + '.1'] = (complex(p, q) / 3) * self.load_mult
                pq_inj[obj['bus']['value'].upper() + '.2'] = (complex(p, q) / 3) * self.load_mult
                pq_inj[obj['bus']['value'].upper() + '.3'] = (complex(p, q) / 3) * self.load_mult
            else:
                pq_inj[obj['bus']['value'].upper() + phaseIdx[obj['phases']['value']]] = complex(p, q) * self.load_mult
                pv_inj[obj['bus']['value'].upper() + phaseIdx[obj['phases']['value']]] = complex(p, q) * 0.0

        for obj in der_pv:
            p = float(obj['p']['value'])
            q = float(obj['q']['value'])
            if 's' in obj['phases']['value']:
                pv_inj[obj['bus']['value'].upper() + phaseIdx[obj['phases']['value']]] = complex(p, q) * self.pv_mult

        Bus = {}
        service_xfmr_bus = {}
        xfmr_buses = []
        phases = []
        for obj in txfmrs:
            xfmr_name = obj['xfmr_name']['value']
            enum = int(obj['end_number']['value'])
            if xfmr_name not in Bus:
                Bus[xfmr_name] = {}
            Bus[xfmr_name][enum] = obj['bus']['value']
            if enum == 1:
                phase = obj['phase']['value']
            if enum == 3:
                xfmr_buses.append(Bus[xfmr_name][1].upper())
                phases.append(phase)

        # Need to store phase information of service transformer bus.
        # Note that multiple service transformers could be connected to same bus in different phase
        for bus in xfmr_buses:
            if bus not in service_xfmr_bus:
                ph_idx = [i for i, val in enumerate(xfmr_buses) if val == bus]
                phase = []
                for p in ph_idx:
                    phase.append(phases[p])
                service_xfmr_bus[bus] = {}
                service_xfmr_bus[bus]['phase'] = phase

        RatedS = {}
        RatedU = {}
        Rohm = {}
        for obj in txfmrs_r:
            xfmr_name = obj['xfmr_name']['value']
            enum = int(obj['enum']['value'])
            if xfmr_name not in RatedS:
                RatedS[xfmr_name] = {}
                RatedU[xfmr_name] = {}
                Rohm[xfmr_name] = {}
            RatedS[xfmr_name][enum] = int(float(obj['ratedS']['value']))
            RatedU[xfmr_name][enum] = int(obj['ratedU']['value'])
            Rohm[xfmr_name][enum] = float(obj['r_ohm']['value'])

        Xohm = {}
        for obj in txfmrs_z:
            xfmr_name = obj['xfmr_name']['value']
            enum = int(obj['enum']['value'])
            # gnum = int(obj['gnum']['value'])
            if xfmr_name not in Xohm:
                Xohm[xfmr_name] = {}
            Xohm[xfmr_name][enum] = float(obj['leakage_z']['value'])

        # Extracting bus information
        s = 0
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
                bus_info[bus]['pq'] = pq_s
                bus_info[bus]['pv'] = pv_s
                idx += 1

        # Extracting branch information
        tpx_xfmr = {}
        idx_line = 0
        for obj in lines:
            bus1, bus2 = obj['bus1']['value'].upper(), obj['bus2']['value'].upper()
            if bus1 in bus_info and bus2 in bus_info:
                if bus_info[bus1]['kv'] == 0.208:
                    line_name = obj['name']['value']
                    tpx_xfmr[line_name] = {}
                    tpx_xfmr[line_name]['idx'] = idx_line
                    tpx_xfmr[line_name]['type'] = 'LINE'
                    ph = [1, 2]
                    tpx_xfmr[line_name]['from'] = bus_info[obj['bus1']['value'].upper()]['idx']
                    tpx_xfmr[line_name]['to'] = bus_info[obj['bus2']['value'].upper()]['idx']
                    tpx_xfmr[line_name]['fr_bus'] = bus1
                    tpx_xfmr[line_name]['to_bus'] = bus2
                    fr_node = []
                    t_node = []
                    for p in ph:
                        from_node = obj['bus1']['value'].upper() + '.' + str(p)
                        to_node = obj['bus2']['value'].upper() + '.' + str(p)
                        fr_node.append(node_name[from_node])
                        t_node.append(node_name[to_node])
                    z = -1 * np.linalg.inv(ybus[np.ix_(fr_node, t_node)])
                    z_prim = np.zeros((2, 2), dtype=complex)
                    # Making the matrix 2 x 2 for triplex line
                    for p in range(len(ph)):
                        for q in range(len(ph)):
                            z_prim[ph[p] - 1][ph[q] - 1] = z[p][q]
                    # tpx_xfmr[line_name]['impedance'] = [0.0042+0.0023j]
                    tpx_xfmr[line_name]['impedance'] = [z[0][0]]
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
                # Store only if the transformer belong to secondary agent. 
                if obj['bus']['value'].upper() in bus_info and txfmrs[idx + 1]['bus']['value'].upper() in bus_info:
                    tpx_xfmr[xmfr_name] = {}
                    tpx_xfmr[xmfr_name]['idx'] = idx_line + idx_xmfr
                    tpx_xfmr[xmfr_name]['type'] = 'SPLIT_PHASE'
                    tpx_xfmr[xmfr_name]['phase'] = obj['phase']['value']
                    tpx_xfmr[xmfr_name]['from'] = bus_info[obj['bus']['value'].upper()]['idx']
                    tpx_xfmr[xmfr_name]['to'] = bus_info[txfmrs[idx + 1]['bus']['value'].upper()]['idx']
                    tpx_xfmr[xmfr_name]['fr_bus'] = obj['bus']['value'].upper()
                    tpx_xfmr[xmfr_name]['to_bus'] = txfmrs[idx + 1]['bus']['value'].upper()

                    # Extracting primary and secondary impedances for split phase transformer
                    zbase1 = RatedU[xmfr_name][1] ** 2 / RatedS[xmfr_name][1]
                    zbase2 = RatedU[xmfr_name][2] ** 2 / RatedS[xmfr_name][2]
                    r1 = Rohm[xmfr_name][1] / zbase1
                    r2 = Rohm[xmfr_name][2] / zbase2
                    r3 = Rohm[xmfr_name][3] / zbase2
                    x12 = Xohm[xmfr_name][1] / zbase1
                    x13 = Xohm[xmfr_name][1] / zbase1
                    x23 = Xohm[xmfr_name][2] / zbase2
                    x1 = 0.5 * (x12 + x13 - x23)
                    x2 = 0.5 * (x12 + x23 - x13)
                    x3 = 0.5 * (x13 + x23 - x12)
                    impedance = [complex(r1, x1)]
                    impedance1 = [complex(r2, x2)]
                    impedance2 = [complex(r3, x3)]
                    tpx_xfmr[xmfr_name]['impedance'] = impedance
                    tpx_xfmr[xmfr_name]['impedance1'] = impedance1
                    idx_xmfr += 1
                    G.add_edge(obj['bus']['value'].upper(), txfmrs[idx + 1]['bus']['value'].upper())

        return bus_info, tpx_xfmr, G, service_xfmr_bus, RatedS

def parse_der_info(der_dict):
    der_info = {}
    for obj in der_dict:
        bus_name = obj['bus']['value']
        der_info[bus_name] = {}
        for key in obj:
            if 'bus' not in key:
                try:
                    der_info[bus_name][key] = float(obj[key]['value'])
                except:
                    der_info[bus_name][key] = obj[key]['value']

    return der_info

def main(feeder_mrid):
    os.environ['GRIDAPPSD_USER'] = "app_user"
    os.environ['GRIDAPPSD_PASSWORD'] = "1234App"
    gapps = GridAPPSD()

    simulation_id = '725830594'
    # feeder_mrid = "_C1C3E687-6FFD-C753-582B-632A27E28507" # IEEE 123 Original
    # feeder_mrid = "_E407CBB6-8C8D-9BC9-589C-AB83FBF0826D" # IEEE 123 PV- NREL
    # feeder_mrid = "_5B816B93-7A5F-B64C-8460-47C17D6E4B0F" # IEEE 13 Assets
    # feeder_mrid = "_49AD8E07-3BF9-A4E2-CB8F-C3722F837B62" # IEEE 13 Node ckt (CDPSM)
    # feeder_mrid = "_59AD8E07-3BF9-A4E2-CB8F-C3722F837B62" # IEEE 123-- New model
    # feeder_mrid = "_AAE94E4A-2465-6F5E-37B1-3E72183A4E44" # IEEE 9500

    model_api_topic = "goss.gridappsd.process.request.data.powergridmodel"

    ######## Query Grid Data #######
    print('Querying grid data \n')
    sparql_mgr = SPARQLManager(gapps, feeder_mrid, model_api_topic, simulation_id)
    ysparse, nodelist = sparql_mgr.ybus_export()
    energy_consumer = sparql_mgr.query_energyconsumer_lf()
    der_pv = sparql_mgr.query_photovoltaic()
    der_batt = sparql_mgr.query_battery()
    lines = sparql_mgr.PerLengthPhaseImpedance_line_names()
    pxfmrs = sparql_mgr.PowerTransformerEnd_xfmr_names()
    txfmrs = sparql_mgr.TransformerTank_xfmr_names()
    switches = sparql_mgr.SwitchingEquipment_switch_names()
    txfmrs_r = sparql_mgr.TransformerTank_xfmr_r()
    txfmrs_z = sparql_mgr.TransformerTank_xfmr_z()
    sourcebus = sparql_mgr.energysource_query()

    ####### Parsing DER Info based on the service bus ########
    energy_consumer_info = parse_der_info(energy_consumer)
    der_pv_info = parse_der_info(der_pv)
    der_batt_info = parse_der_info(der_batt)

    ###### Assigning Load & DER Multiplier to simulate time steps ######
    load_mult = 0.25
    pv_mult = 1.0

    ###### Node list into dictionary ######
    node_name = {}
    for idx, obj in enumerate(nodelist):
        node_name[obj.strip('\"')] = idx
    cnv = sparql_mgr.vnom_export()
    N = len(node_name)
    ybus = np.zeros((N, N), dtype=complex)
    for obj in ysparse:
        items = obj.split(',')
        if items[0] == 'Row':  # skip header line
            continue
        ybus[int(items[0]) - 1][int(items[1]) - 1] = ybus[int(items[1]) - 1][int(items[0]) - 1] = complex(
            float(items[2]), float(items[3]))

    print('Extracting agent-specific data \n')
    agent_input = AgentData(ybus, cnv, node_name, energy_consumer, der_pv, lines, pxfmrs, txfmrs, txfmrs_r, txfmrs_z,
                            switches, load_mult, pv_mult)


    # Extracting area agent (network model other than service level) from the grid data
    bus_info, branch_sw_data, G = agent_input.area_agent()
    bus_info_inj = copy.deepcopy(bus_info)

    # Extracting secondary agents' data from the grid data
    bus_info_sec, tpx_xfmr, G_sec, service_xfmr_bus, RatedS = agent_input.secondary_agent()

    # Initialize secondary transformer agents
    alpha_store = {}

    Secondary_agent_list= {}
    for agent_bus in service_xfmr_bus:
        for p in service_xfmr_bus[agent_bus]['phase']:
            # If there are multiple service transformers in a bus, we need to separate them
            G_sec_p = copy.deepcopy(G_sec)
            if len(service_xfmr_bus[agent_bus]['phase']) > 1:
                for e in tpx_xfmr:
                    if tpx_xfmr[e]['type'] == 'SPLIT_PHASE' and tpx_xfmr[e]['phase'] != p:
                        G_sec_p.remove_edge(tpx_xfmr[e]['fr_bus'], tpx_xfmr[e]['to_bus'])

            sp_conn_graph = list(nx.connected_components(G_sec_p))
            Secondary_agent_list[agent_bus+p] = Secondary_Agent(agent_bus, sp_conn_graph, bus_info_sec, tpx_xfmr, p, sparql_mgr.feeder_mrid, energy_consumer_info, der_pv_info, der_batt_info)
            ## Intializing Alpha storing variable
            alpha_store[agent_bus+p] = {}

    # Initialize the voltage for service transformer to be 1.04 pu
    bus_voltage = {}
    bus_voltage_plot = {}
    xfmr_bus = []
    for agent_bus in service_xfmr_bus:
        xfmr_bus.append(agent_bus)
        # alpha_store[agent_bus] = {}
        bus_voltage[agent_bus] = {}
        bus_voltage[agent_bus]['A'] = [1.04]
        bus_voltage[agent_bus]['B'] = [1.04]
        bus_voltage[agent_bus]['C'] = [1.04]
        bus_voltage_plot[agent_bus] = []

    # Starting the iteration and message exchange
    count = 0
    iter = 30
    error = 1
    s_inj = []
    v_substation = [1.04, 1.04, 1.04]

    while 1:
        ########################### SECONDARY AGENTS #################################
        # Extract the inputs required for each secondary agents
        # Store the equivalent injection to pass into area agent
        message_injection = {}
        err = [0]
        print("\nInvoking the service transformer agents. Iteration count = ", count)
        for SA_id in Secondary_agent_list:
            SA_instance = Secondary_agent_list[SA_id]
            agent_bus = SA_instance.name

            if agent_bus not in  message_injection:
                message_injection[agent_bus] = [0j, 0j, 0j]

            ratedS = RatedS[SA_instance.xfmr_name][1]
            agent_bus_idx = SA_instance.bus_info_sec_agent_i[agent_bus]['idx']
            # Source bus is a transformer primary.
            # Trying different moving average to avoid oscillation in PCC
            guess = 6
            if count < guess:
                vsrc = [sum(bus_voltage[agent_bus][SA_instance.phase]) / len(bus_voltage[agent_bus][SA_instance.phase])]
            else:
                vsrc = [sum(bus_voltage[agent_bus][SA_instance.phase][-(guess-1):]) / len(bus_voltage[agent_bus][SA_instance.phase][-(guess-1):])]

            # Invoking the optimization
            sec_inj, alpha, sec_bus_voltage = SA_instance.alpha_area(agent_bus, agent_bus_idx, vsrc, service_xfmr_bus, SA_instance.phase, ratedS)
            message_injection[agent_bus] = np.add(message_injection[agent_bus], sec_inj).tolist()
            alpha_store[SA_id][count] = alpha

            if count >= 1:
                err.append(abs(alpha_store[SA_id][count] - alpha_store[SA_id][count - 1]))

        if count >= 1:
            error = max(err)

        ########################### COORDINATING AGENT ###############################
        for k in message_injection:
            bus_info[k]['injection'] = message_injection[k]

        s = 0
        for k in bus_info:
            s += sum(bus_info[k]['injection']) / 1000.0
        s_inj.append(s)
        # Finding the switch delimited areas and give the area specific information to agents    
        # edge = [['18', '135'], ['151', '300_OPEN']]
        # edge = [['60', '160'], ['97', '197'], ['54', '94_OPEN']]
        # edge = [['151', '300'], ['97', '197']]
        # sourcebus = '150'
        # branch_sw_data_area_i, bus_info_area_i = area_info(G, edge, branch_sw_data, bus_info, sourcebus)

        print('\nInvoking coordinating agent')
        area_i_agent = AreaCoordinator()
        agent_bus = sourcebus[0]['bus']['value'].upper()
        agent_bus_idx = bus_info[agent_bus]['idx']
        bus_voltage_area = area_i_agent.alpha_area(branch_sw_data, bus_info, agent_bus, agent_bus_idx, v_substation, service_xfmr_bus)

        # If no secondary network exists, no need to iterate on the optimization
        if len(service_xfmr_bus) == 0:
            print('No secondary network in the feeder model. Terminating without optimization')
            exit()

        # Extract voltage at buses from the area agent optimization
        for k in bus_voltage_area:
            try:
                bus_voltage[k]['A'].append(bus_voltage_area[k]['A'])
                bus_voltage[k]['B'].append(bus_voltage_area[k]['B'])
                bus_voltage[k]['C'].append(bus_voltage_area[k]['C'])
            except:
                continue

        count += 1
        # Check number of iterations and maximum error in alpha convergence
        if count >= iter or error < 0.001:
            phaseA = []
            phaseB = []
            phaseC = []
            for k in bus_voltage_area:
                if bus_voltage_area[k]['A'] > 0.5:
                    phaseA.append(bus_voltage_area[k]['A'])
                if bus_voltage_area[k]['B'] > 0.5:
                    phaseB.append(bus_voltage_area[k]['B'])
                if bus_voltage_area[k]['C'] > 0.5:
                    phaseC.append(bus_voltage_area[k]['C'])
            break

    ###### Assigning converged alpha to HEMS DERs ######
    for SA_id in Secondary_agent_list:
        SA_instance = Secondary_agent_list[SA_id]
        f_name = '../outputs/agent_' + SA_id + '.json'
        service_xfmr_agent = {}
        service_xfmr_agent[SA_id] = {}
        json_fp = open(f_name, 'w')
        for HEMS in SA_instance.HEMS_list:
            HEMS_instance = SA_instance.HEMS_list[HEMS]
            if HEMS_instance.PV:
                HEMS_instance.PV['set_point'] = HEMS_instance.PV['p']* pv_mult * (1 - alpha_store[SA_id][count - 1])
                pv_diff = DifferenceBuilder(simulation_id)
                pv_diff.add_difference(HEMS_instance.PV['id'], "PowerElectronicsConnection.p", HEMS_instance.PV['set_point'], 0)
                msg = pv_diff.get_message()
                service_xfmr_agent[SA_id][HEMS_instance.PV['name']] = msg
        json.dump(service_xfmr_agent, json_fp, indent=2)
        json_fp.close()


    ###### Plot the voltage after the convergence ######
    plt.plot(phaseA)
    plt.plot(phaseB)
    plt.plot(phaseC)
    plt.xlabel('Nodes')
    plt.ylabel('Voltage (p.u.)')
    plt.legend(['Phase A', 'Phase B', 'Phase C'])
    plt.grid()
    plt.show()

    ###### Plot individual alphas for service XFMRs after the convergence ######
    for SA_id in Secondary_agent_list:
        alpha = []
        for k in range(count):
            alpha.append(alpha_store[SA_id][k])
        plt.plot(alpha[-(count-1):])
        # print(agent_bus, alpha)

    plt.xlabel('Iterations')
    plt.ylabel('PV power curtailment factor (alpha)')
    plt.grid()
    plt.show()

    ###### plot the voltage convergence as well ######
    for agent_bus in service_xfmr_bus:
        plt_v = len(bus_voltage[agent_bus]['A'])
        # plt.plot(bus_voltage[agent_bus]['A'][-(plt_v-1):])
        # plt.plot(bus_voltage[agent_bus]['B'][-(plt_v-1):])
        plt.plot(bus_voltage[agent_bus]['A'][-(plt_v-1):])

    plt.xlabel('Iterations')
    plt.ylabel('PCC Voltages for service transformers (p.u.)')
    plt.grid()
    plt.show()
    plt.plot(s_inj)
    plt.xlabel('Iterations')
    plt.ylabel('Total injections from service transformers (kW)')
    plt.grid()
    plt.show()


if __name__ == '__main__':

    if len(sys.argv) > 1:
        feeder_mrid = sys.argv[1]
        main(feeder_mrid)
    else:
        print("No feeder mrid was provided while invoking the application")
