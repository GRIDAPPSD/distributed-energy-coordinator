import os
import json
import logging
from typing import Dict
from gridappsd.field_interface.agents import CoordinatingAgent, FeederAgent, SwitchAreaAgent, SecondaryAreaAgent
from gridappsd.field_interface.interfaces import MessageBusDefinition
from simulation import Sim
import auth

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

BUS_CONFIG = os.environ.get('BUS_CONFIG')
OUTPUT_DIR = os.environ.get('OUTPUTS')

class SampleCoordinatingAgent(CoordinatingAgent):
    def __init__(self, feeder_id, system_message_bus_def, simulation_id=None):
        super().__init__(feeder_id, system_message_bus_def, simulation_id)


class SampleFeederAgent(FeederAgent):
    def __init__(self, upstream: MessageBusDefinition, downstream: MessageBusDefinition, feeder: Dict = None, simulation_id: str = None) -> None:
        super().__init__(upstream, downstream, feeder, simulation_id)
        self._latch = False

    def on_measurement(self, headers: Dict, message) -> None:
        if not self._latch:
            log.debug("measurement: %s.%s", self.__class__.__name__, headers.get('destination'), exc_info=True)
            with open(f'{OUTPUT_DIR}/feeder.json', "w", encoding='UTF-8') as file:
                file.write(json.dumps(message))
            self._latch = True


class SampleSwitchAreaAgent(SwitchAreaAgent):
    def __init__(self, upstream: MessageBusDefinition, downstream: MessageBusDefinition, feeder: Dict = None, simulation_id: str = None) -> None:
        super().__init__(upstream, downstream, feeder, simulation_id)
        self._latch = False

    def on_measurement(self, headers: Dict, message):
        if not self._latch:
            log.debug("measurement: %s.%s", self.__class__.__name__, headers.get('destination'), exc_info=True)
            with open(f'{OUTPUT_DIR}/switch_area.json', "w", encoding='UTF-8') as file:
                file.write(json.dumps(message))
            self._latch = True

class SampleSecondaryAreaAgent(SecondaryAreaAgent):
    def __init__(self, upstream: MessageBusDefinition, downstream: MessageBusDefinition, feeder: Dict = None, simulation_id: str = None) -> None:
        super().__init__(upstream, downstream, feeder, simulation_id)
        self._latch = False

    def on_measurement(self, headers: Dict, message):
        if not self._latch:
            log.debug("measurement: %s.%s", self.__class__.__name__, headers.get('destination'), exc_info=True)
            with open(f'{OUTPUT_DIR}/secondary.json', "w", encoding='UTF-8') as file:
                file.write(json.dumps(message))
            self._latch = True

def overwrite_parameters(feeder_id: str, area_id: str = '') -> MessageBusDefinition:
    """_summary_
    """
    bus_def = MessageBusDefinition.load(BUS_CONFIG)
    if area_id:
        bus_def.id = feeder_id + '.' + area_id
    else:
        bus_def.id = feeder_id

    address = os.environ.get('GRIDAPPSD_ADDRESS')
    port = os.environ.get('GRIDAPPSD_PORT')
    if not address or not port:
        raise ValueError("import auth_context or set environment up before this statement.")

    bus_def.conneciton_args['GRIDAPPSD_ADDRESS'] = f"tcp://{address}:{port}"
    bus_def.conneciton_args['GRIDAPPSD_USER'] = os.environ.get('GRIDAPPSD_USER')
    bus_def.conneciton_args['GRIDAPPSD_PASSWORD'] = os.environ.get('GRIDAPPSD_PASSWORD')
    return bus_def

def save_area(context: dict) -> None:
    """_summary_
    """
    with open(f"{OUTPUT_DIR}/{context['message_bus_id']}.json", "w", encoding='UTF-8') as file:
        file.write(json.dumps(context))
        
def spawn_feeder_agent(context: dict, sim: Sim, coord_agent: SampleCoordinatingAgent) -> None:
    sys_message_bus = overwrite_parameters(sim.get_feeder_id())
    feeder_message_bus = overwrite_parameters(sim.get_feeder_id())
    save_area(context)
    agent = SampleFeederAgent(sys_message_bus,
                              feeder_message_bus,
                              context,
                              sim.get_simulation_id())
    coord_agent.spawn_distributed_agent(agent)
    
def spawn_secondary_agents(context: dict, sw_idx: int, sec_idx: int, sim: Sim, coord_agent: SampleCoordinatingAgent) -> None:
        switch_message_bus = overwrite_parameters(sim.get_feeder_id(), f"{sw_idx}")
        secondary_area_message_bus_def = overwrite_parameters(sim.get_feeder_id(), f"{sw_idx}.{sec_idx}")
        save_area(context)
        agent = SampleSecondaryAreaAgent(switch_message_bus,
                                         secondary_area_message_bus_def,
                                         context,
                                         sim.get_simulation_id())
        if len(agent.secondary_area.addressable_equipment) > 1:
            coord_agent.spawn_distributed_agent(agent)
            
def spawn_switch_area_agents(context: dict, idx: int, sim: Sim, coord_agent: SampleCoordinatingAgent) -> None:
    feeder_message_bus = overwrite_parameters(sim.get_feeder_id())
    save_area(context)
    switch_message_bus = overwrite_parameters(sim.get_feeder_id(), f"{idx}")
    agent = SampleSwitchAreaAgent(feeder_message_bus,
                                  switch_message_bus,
                                  context,
                                  sim.get_simulation_id())
    coord_agent.spawn_distributed_agent(agent)
    
    # Get all the attributes of the equipments in the switch area from the model 
    # EXAMPLE 1 - Get phase, bus info about ACLineSegments
    # example.get_lines_buses(agent.switch_area)
    
    # # EXAMPLE 2 - Get all line impedance data
    # example.get_line_impedances(agent.switch_area)
    
    # # EXAMPLE 3 - Sort all line impedance by line phase:
    # example.sort_impedance_by_line(agent.switch_area)
    
    # # Example 4 - Sort all lines by impedance
    # example.sort_line_by_impedance(agent.switch_area)
    
    # # Example 5 - Get TransformerTank impedances
    # example.get_tank_impedances(agent.switch_area)
    
    # # Example 6 - Get inverter buses and phases
    # example.get_inverter_buses(agent.switch_area)
    
    # # Example 7 - Get load buses and phases
    # example.get_load_buses(agent.switch_area)