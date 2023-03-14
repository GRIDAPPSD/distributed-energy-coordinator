import os
import logging
import time
import json
from simulation import Sim
import agent
import auth

from gridappsd import GridAPPSD
import gridappsd.field_interface.agents.agents as agents_mod
from gridappsd.field_interface.agents import CoordinatingAgent, FeederAgent, SwitchAreaAgent, SecondaryAreaAgent
from gridappsd.field_interface.context import ContextManager
from gridappsd.field_interface.interfaces import MessageBusDefinition
from cimlab.data_profile import CIM_PROFILE
cim_profile = CIM_PROFILE.RC4_2021.value
agents_mod.set_cim_profile(cim_profile)
cim = agents_mod.cim

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())
log.setLevel(logging.DEBUG)

def load_json(path:str) -> dict:
    with open(path, 'r', encoding='UTF-8') as file:
        return json.load(file)

def load_cfg(path:str) -> str:
    with open(path, 'r') as file:
        return file.read()

def compare_config(goss_config: str, sim_config: dict) -> None:
    sim_feeder = sim_config["power_system_config"]["Line_name"]
    assert sim_feeder in goss_config, f"file {goss_config} field.model.mrid does not match {sim_feeder}"
    
def run() -> None:
    goss_config = load_cfg(os.environ.get('GOSS_CONFIG'))
    sim_config = load_json(os.environ.get('SIM_CONFIG'))
    compare_config(goss_config, sim_config)

    sim = Sim(sim_config)
    
    sys_message_bus = agent.overwrite_parameters(sim.get_feeder_id())

    coordinating_agent = agent.SampleCoordinatingAgent(sim.get_feeder_id(), sys_message_bus)
    
    context = ContextManager.get_context_by_feeder(sim.get_feeder_id())
    
    feeder = context['data']
    agent.spawn_feeder_agent(feeder, sim, coordinating_agent)
    
    for sw_idx, switch_area in enumerate(feeder['switch_areas']):
        agent.spawn_switch_area_agents(switch_area, sw_idx, sim, coordinating_agent)
        
        for sec_index, secondary_area in enumerate(switch_area['secondary_areas']):
            agent.spawn_secondary_agents(secondary_area, sw_idx, sec_index, sim, coordinating_agent)

    while not sim.done:
        try:
            time.sleep(0.1)
        except KeyboardInterrupt:
            print("Exiting sample")
            break
