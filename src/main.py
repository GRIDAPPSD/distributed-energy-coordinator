import traceback
import os
import logging
import time
import json
from enum import Enum
from sim import Sim
from agents import overwrite_parameters
from agents import SampleFeederAgent
from agents import SampleCoordinatingAgent
from agents import SampleSwitchAreaAgent
from agents import SampleSecondaryAreaAgent
from context import FeederAreaContextManager
from context import SwitchAreaContextManager
from context import SecondaryAreaContextManager

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class LineName(Enum):
    IEEE13 = "_49AD8E07-3BF9-A4E2-CB8F-C3722F837B62"
    IEEE123 = "_C1C3E687-6FFD-C753-582B-632A27E28507"


def initialize():
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ['OUTPUT_DIR'] = f"{ROOT}/outputs"
    os.environ['BUS_CONFIG'] = f"{ROOT}/config/system_message_bus.yml"
    os.environ['GOSS_CONFIG'] = f"{ROOT}/config/pnnl.goss.gridappsd.cfg"
    os.environ['SIM_CONFIG'] = f"{ROOT}/config/ieee13.json"
    os.environ['GRIDAPPSD_APPLICATION_ID'] = 'dist-admm'
    os.environ['GRIDAPPSD_USER'] = 'system'
    os.environ['GRIDAPPSD_PASSWORD'] = 'manager'
    os.environ['GRIDAPPSD_ADDRESS'] = 'localhost'
    os.environ['GRIDAPPSD_PORT'] = '61613'


def load_json(path: str) -> dict:
    with open(path, 'r', encoding='UTF-8') as file:
        return json.load(file)


def load_cfg(path: str) -> str:
    with open(path, 'r') as file:
        return file.read()


def compare_config(goss_config: str, sim_config: dict) -> None:
    sim_feeder = sim_config["power_system_config"]["Line_name"]
    assert sim_feeder in goss_config, f"file {goss_config} field.model.mrid does not match {sim_feeder}"


def start_simulation(config: dict) -> Sim:
    log.debug(
        f"Simulation for feeder: {config['power_system_config']['Line_name']}")
    return Sim(config)


def spawn_context_managers(sim: Sim) -> None:

    system_bus = overwrite_parameters(sim.get_feeder_id())

    config = {
        "app_id": "context_manager",
        "description": "This agent provides topological context information like neighboring agents and devices to other distributed agents"
    }

    feeder_bus = system_bus
    feeder_manager = FeederAreaContextManager(
        system_bus, feeder_bus, config, None, sim.get_simulation_id())

    switch_areas = feeder_manager.agent_area_dict['switch_areas']
    log.debug(switch_areas)
    for sw_idx, switch_area in enumerate(switch_areas):
        switch_bus = overwrite_parameters(sim.get_feeder_id(), f"{sw_idx}")
        if sw_idx != 0:
            switch_manager = SwitchAreaContextManager(
                feeder_bus, switch_bus, config, switch_area, sim.get_simulation_id())

        secondary_areas = switch_area['secondary_areas']
        log.debug(secondary_areas)
        for sec_idx, secondary_area in enumerate(secondary_areas):
            secondary_bus = overwrite_parameters(
                sim.get_feeder_id(), f"{sw_idx}.{sec_idx}")
            secondary_manager = SecondaryAreaContextManager(
                switch_bus, secondary_bus, config, secondary_area, sim.get_simulation_id())


def spawn_agents(sim: Sim) -> None:

    system_bus = overwrite_parameters(sim.get_feeder_id())

    coordinating_agent = SampleCoordinatingAgent(
        system_bus, sim.get_simulation_id())

    config = {
        "app_id": "sample_app",
        "description":
            "This is a GridAPPS-D sample distribution application agent"
    }

    feeder_bus = system_bus
    feeder_agent = SampleFeederAgent(
        system_bus, feeder_bus, config, None, sim.get_simulation_id())
    
    switch_areas = feeder_agent.agent_area_dict['switch_areas']
    log.debug(switch_areas)
    for sw_idx, switch_area in enumerate(switch_areas):
        switch_bus = overwrite_parameters(sim.get_feeder_id(), f"{sw_idx}")
        if sw_idx != 0:
            switch_agent = SampleSwitchAreaAgent(
                feeder_bus, switch_bus, config, switch_area, sim.get_simulation_id())
            
        secondary_areas = switch_area['secondary_areas']
        log.debug(secondary_areas)
        for sec_idx, secondary_area in enumerate(secondary_areas):
            secondary_bus = overwrite_parameters(
                sim.get_feeder_id(), f"{sw_idx}.{sec_idx}")
            secondary_agent = SampleSecondaryAreaAgent(
                switch_bus, secondary_bus, config, secondary_area, sim.get_simulation_id())


def run():

    try:

        initialize()
        sim_config = load_json(os.environ.get('SIM_CONFIG'))
        goss_config = load_cfg(os.environ.get('GOSS_CONFIG'))
        compare_config(goss_config, sim_config)

        sim = start_simulation(sim_config)
        spawn_context_managers(sim)
        spawn_agents(sim)

    except Exception as e:
        log.debug(e)
        log.debug(traceback.format_exc())
        sim.simulation.stop()

    while not sim.done:
        try:
            time.sleep(0.1)
        except KeyboardInterrupt:
            log.debug("Exiting sample")
            sim.simulation.stop()
            break


if __name__ == "__main__":
    run()
