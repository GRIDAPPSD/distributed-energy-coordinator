import os
import json
import logging
from typing import Dict, List, Tuple
from gridappsd.field_interface.agents import CoordinatingAgent, FeederAgent, SwitchAreaAgent, SecondaryAreaAgent
from gridappsd.field_interface.interfaces import MessageBusDefinition
from simulation import Sim
import queries as qy

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())
log.setLevel(logging.DEBUG)

BUS_CONFIG = os.environ.get('BUS_CONFIG')
OUTPUT_DIR = os.environ.get('OUTPUTS')

class SampleCoordinatingAgent(CoordinatingAgent):
    def __init__(self, feeder_id, system_message_bus_def, simulation_id=None):
        super().__init__(feeder_id, system_message_bus_def, simulation_id)

class SampleFeederAgent(FeederAgent):
    
    def __init__(self, upstream: MessageBusDefinition, downstream: MessageBusDefinition, feeder: Dict = None, simulation_id: str = None) -> None:
        super().__init__(upstream, downstream, feeder, simulation_id)
        self._latch = False
        # init_cim(self.feeder_area)
        # self.line_info, self.bus_info = query_line_info(self.feeder_area)
        # print(self.bus_info)

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
        qy.init_cim(self.switch_area)
        self.line_info, self.bus_info = qy.query_line_info(self.switch_area)
        self.bus_info.update(qy.query_transformers(self.switch_area))
        self.bus_info.update(qy.query_power_electronics(self.switch_area))
        self.bus_info.update(qy.query_energy_consumers(self.switch_area))
        
        save_info(f'{feeder["message_bus_id"]}_lineinfo', self.line_info)
        save_info(f'{feeder["message_bus_id"]}_businfo', self.bus_info)


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
        qy.init_cim(self.secondary_area)
        self.line_info, self.bus_info = qy.query_line_info(self.secondary_area)

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
    with open(f"{OUTPUT_DIR}/{context['message_bus_id']}.json", "w", encoding='UTF-8') as file:
        file.write(json.dumps(context))
        
def save_info(context: str, info: dict) -> None:
    with open(f"{OUTPUT_DIR}/{context}.json", "w", encoding='UTF-8') as file:
        file.write(json.dumps(info))

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