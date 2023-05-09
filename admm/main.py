import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['OUTPUTS'] = f"{ROOT}/outputs"
os.environ['BUS_CONFIG'] = f"{ROOT}/config/system_message_bus.yml"
os.environ['GOSS_CONFIG'] = f"{ROOT}/config/pnnl.goss.gridappsd.cfg"
os.environ['SIM_CONFIG'] = f"{ROOT}/config/ieee123.json"
os.environ['GRIDAPPSD_APPLICATION_ID'] = 'dist-sample-app'
os.environ['GRIDAPPSD_USER'] = 'app_user'
os.environ['GRIDAPPSD_PASSWORD'] = '1234App'
os.environ['GRIDAPPSD_ADDRESS'] = 'localhost'
os.environ['GRIDAPPSD_PORT'] = '61613'

import logging
import time
import json
from simulation import Sim
import agent
from gridappsd.field_interface.context import ContextManager

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
        if sw_idx != 0:
            agent.spawn_switch_area_agents(switch_area, sw_idx, sim, coordinating_agent)
        
        for sec_index, secondary_area in enumerate(switch_area['secondary_areas']):
            agent.spawn_secondary_agents(secondary_area, sw_idx, sec_index, sim, coordinating_agent)

    while not sim.done:
        try:
            time.sleep(0.1)
        except KeyboardInterrupt:
            print("Exiting sample")
            sim.simulation.stop()
            break

if __name__ == "__main__":
    run()