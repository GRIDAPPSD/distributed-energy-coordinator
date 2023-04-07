import os
from typing import Tuple
import numpy as np
import json
import gridappsd.field_interface.agents.agents as agents_mod
from cimlab.data_profile import CIM_PROFILE
cim_profile = CIM_PROFILE.RC4_2021.value
agents_mod.set_cim_profile(cim_profile)
cim = agents_mod.cim

OUTPUT_DIR = os.environ.get('OUTPUTS')

def log (context: str, data) -> None:
    with open(f"{OUTPUT_DIR}/{context}.json", "w", encoding='UTF-8') as file:
        file.write(str(data))

def init_cim(network_area) -> None:
    network_area.get_all_attributes(cim.ACLineSegment)
    network_area.get_all_attributes(cim.ACLineSegmentPhase)
    network_area.get_all_attributes(cim.PerLengthPhaseImpedance)
    network_area.get_all_attributes(cim.PhaseImpedanceData)
    network_area.get_all_attributes(cim.WireSpacingInfo)
    network_area.get_all_attributes(cim.WirePosition)
    network_area.get_all_attributes(cim.OverheadWireInfo)
    network_area.get_all_attributes(cim.ConcentricNeutralCableInfo)
    network_area.get_all_attributes(cim.TapeShieldCableInfo)
    network_area.get_all_attributes(cim.TransformerTank)
    network_area.get_all_attributes(cim.TransformerTankEnd)
    network_area.get_all_attributes(cim.TransformerTankInfo)
    network_area.get_all_attributes(cim.TransformerEndInfo)
    network_area.get_all_attributes(cim.ShortCircuitTest)
    network_area.get_all_attributes(cim.NoLoadTest)
    network_area.get_all_attributes(cim.PowerElectronicsConnection)
    network_area.get_all_attributes(cim.PowerElectronicsConnectionPhase)    
    network_area.get_all_attributes(cim.EnergyConsumer)
    network_area.get_all_attributes(cim.EnergyConsumerPhase)
    network_area.get_all_attributes(cim.Terminal)

def query_line_info(network_area) -> Tuple[dict, dict]:
    if cim.ACLineSegment not in network_area.typed_catalog:
            return {}, {}
        
    line_info = {}
    bus_info = {}
    bus_id = 0
    line_ids = list(network_area.typed_catalog[cim.ACLineSegment].keys())

    for i, line_id in enumerate(line_ids):
        line = network_area.typed_catalog[cim.ACLineSegment][line_id]
        log(line.name, line)
        line_info[line.name] = {}
        line_info[line.name]['idx'] = i
        line_info[line.name]['type'] = 'LINE'

        from_bus = line.Terminals[0].ConnectivityNode.name
        if from_bus not in bus_info:
            bus_info[from_bus] = bus_id
            bus_id += 1
        line_info[line.name]['fr_bus'] = from_bus
        line_info[line.name]['from'] = bus_info[from_bus]
            
        to_bus = line.Terminals[1].ConnectivityNode.name
        if to_bus not in bus_info:
            bus_info[to_bus] = bus_id
            bus_id += 1
        line_info[line.name]['to_bus'] = to_bus
        line_info[line.name]['to'] = bus_info[to_bus]

        if line.PerLengthImpedance.PhaseImpedanceData is not None:
            length = float(line.length)
            phase_count = 3
            line_info[line.name]['zprim'] = [[[0.0,0.0] for _ in range(phase_count)] for _ in range(phase_count)]
            for data in line.PerLengthImpedance.PhaseImpedanceData:
                log(line.name+'zprim', data)
                row = int(data.row)-1
                col = int(data.column)-1
                real = length*float(data.r)
                imag = length*float(data.x)
                line_info[line.name]['zprim'][row][col] = [real,imag]
                line_info[line.name]['zprim'][col][row] = [real,imag]
                
        line_info[line.name]['phases'] = []
        for line_phs in line.ACLineSegmentPhases:
            log(line.name+line_phs.phase[0], line_phs)
            line_info[line.name]['phases'].append(line_phs.phase[0])
            
    return line_info, bus_info

def query_transformers(network_area) -> dict:
    if cim.TransformerTank not in network_area.typed_catalog:
        return {}
    
    bus_info = {}
    
    for tank in network_area.typed_catalog[cim.TransformerTank].values():
        log(tank.name, tank)
        for end in tank.TransformerTankEnds:
            node = end.Terminal.ConnectivityNode
            bus_info[node.name] = {}
            
            for end_info in tank.TransformerTankInfo.TransformerEndInfos:
                bus_info[node.name]['kv'] = end_info.ratedU
    
    return bus_info
        
def query_power_electronics(network_area) -> dict:
    if cim.PowerElectronicsConnection not in network_area.typed_catalog:
        return {}
    
    bus_info = {}
    
    for pec in network_area.typed_catalog[cim.PowerElectronicsConnection].values():
        node = pec.Terminals[0].ConnectivityNode
        bus_info[node.name] = {}
        log(pec.name, pec)
        
        phase_count = 3
        bus_info[node.name]['pq'] = [[0.0,0.0] for _ in range(phase_count)]
        for pec_phs in pec.PowerElectronicsConnectionPhases:
            if pec_phs.phase[0] == 'A':
                bus_info[node.name]['pq'][0] = [pec.p, pec.q]
                
            if pec_phs.phase[0] == 'B':
                bus_info[node.name]['pq'][1] = [pec.p, pec.q]
                
            if pec_phs.phase[0] == 'C':
                bus_info[node.name]['pq'][2] = [pec.p, pec.q]
                
    return bus_info
            
#sort EnergyConsumers
def query_energy_consumers(network_area) -> dict:
    if cim.EnergyConsumer not in network_area.typed_catalog:
        return {}

    bus_info = {}
    
    for load in network_area.typed_catalog[cim.EnergyConsumer].values():
        node = load.Terminals[0].ConnectivityNode
        bus_info[node.name] = {}
        log(load.name, load)
        
        phase_count = 3
        bus_info[node.name]['pq'] = [[0.0,0.0] for _ in range(phase_count)]
        for load_phs in load.EnergyConsumerPhase:
            if load_phs.phase:
                if load_phs.phase[0] == 'A':
                    bus_info[node.name]['pq'][0] = [load_phs.p, load_phs.q]
                    
                if load_phs.phase[0] == 'B':
                    bus_info[node.name]['pq'][1] = [load_phs.p, load_phs.q]
                    
                if load_phs.phase[0] == 'C':
                    bus_info[node.name]['pq'][2] = [load_phs.p, load_phs.q]
                
    return bus_info