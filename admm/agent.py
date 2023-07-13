import os
import json
import logging
from collections import OrderedDict
from typing import Dict
import numpy as np
from gridappsd.field_interface.agents import (
    CoordinatingAgent,
    FeederAgent,
    SwitchAreaAgent,
    SecondaryAreaAgent,
)
from gridappsd.field_interface.interfaces import MessageBusDefinition
from simulation import Sim
import queries as qy
from alpha_area import AlphaArea

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())
log.setLevel(logging.DEBUG)

SOURCE_BUSES = {"4": "149", "5": "135", "3": "152", "1": "160", "2": "197"}
SOURCE_VOLTAGE = [1.0475, 1.0475, 1.0475]
ALPHAS = [0, 0, 0, 0, 0]
LAMBDA_V = np.zeros((5, 3))
LAMBDA = [0, 0, 0, 0, 0]
LAMBDA_P = [0, 0, 0]
LAMBDA_Q = [0, 0, 0]
MU_V_ALPHA = [7000, 1000000, 1000000, 5]
CHILD = []
LAST_P = [0, 0, 0]
LAST_Q = [0, 0, 0]
LAST_V = [0, 0, 0]


class SampleCoordinatingAgent(CoordinatingAgent):
    def __init__(self, feeder_id, system_message_bus_def, simulation_id=None):
        super().__init__(system_message_bus_def, simulation_id)


class SampleFeederAgent(FeederAgent):
    def __init__(
        self,
        upstream: MessageBusDefinition,
        downstream: MessageBusDefinition,
        config: Dict,
        feeder: Dict = None,
        simulation_id: str = None,
    ) -> None:
        super().__init__(upstream, downstream, config, feeder, simulation_id)
        self._latch = False
        self.connect()
        # init_cim(self.feeder_area)
        # self.line_info, self.bus_info = query_line_info(self.feeder_area)
        # print(self.bus_info)

    def on_measurement(self, headers: Dict, message) -> None:
        if not self._latch:
            log.debug(
                "measurement: %s.%s",
                self.__class__.__name__,
                headers.get("destination"),
                exc_info=True,
            )
            with open(f"{os.environ.get('OUTPUT_DIR')}/feeder.json", "w", encoding="UTF-8") as file:
                file.write(json.dumps(message))
            self._latch = True

    def on_downstream_message(self, headers: Dict, message) -> None:
        print(headers)
        print(message)

    def on_upstream_message(self, headers: Dict, message) -> None:
        print(headers)
        print(message)


class SampleSwitchAreaAgent(SwitchAreaAgent):
    def __init__(
        self,
        upstream: MessageBusDefinition,
        downstream: MessageBusDefinition,
        config: Dict,
        feeder: Dict = None,
        simulation_id: str = None,
    ) -> None:
        super().__init__(upstream, downstream, config, feeder, simulation_id)
        self._latch = False
        self.area = feeder["message_bus_id"][-1:]
        self.alpha = AlphaArea()
        qy.init_cim(self.switch_area)
        self.branch_info, self.bus_info = qy.query_line_info(self.switch_area)

        branch, bus = qy.query_transformers(self.switch_area)
        self.branch_info.update(branch)
        self.bus_info.update(bus)

        self.bus_info.update(qy.query_power_electronics(self.switch_area))
        self.bus_info.update(qy.query_energy_consumers(self.switch_area))

        self.branch_info, self.bus_info = qy.index_info(
            self.branch_info, self.bus_info)

        print(f'branch count: {len(self.branch_info.keys())}')
        print(f'bus count:  {len(self.bus_info.keys())}')

        save_info(
            f'{feeder["message_bus_id"]}_lineinfo',
            OrderedDict(sorted(self.branch_info.items())),
        )
        save_info(
            f'{feeder["message_bus_id"]}_businfo',
            OrderedDict(sorted(self.bus_info.items())),
        )

        self.alpha.alpha_area(
            self.branch_info,
            self.bus_info,
            SOURCE_BUSES[self.area],
            self.bus_info[SOURCE_BUSES[self.area]]["idx"],
            SOURCE_VOLTAGE,
            0,
            True,
            int(self.area),
            ALPHAS,
            LAMBDA_V,
            LAMBDA,
            LAMBDA_P,
            LAMBDA_Q,
            MU_V_ALPHA,
            CHILD,
            LAST_P,
            LAST_Q,
            LAST_V
        )

        self.connect()

    def on_measurement(self, headers: Dict, message):
        message = {
            "area": self.area,
            "message": "whatever",
            "tags": ["area", "any_common_key"]
        }
        self.switch_area.publish_upstream(message)
        if not self._latch:
            log.debug(
                "measurement: %s.%s",
                self.__class__.__name__,
                headers.get("destination"),
                exc_info=True,
            )
            with open(f"{os.environ.get('OUTPUT_DIR')}/switch_area.json", "w", encoding="UTF-8") as file:
                file.write(json.dumps(message))

            self._latch = True
            self.downstream_message_bus.publish(self.bus_info, self.area)

    def on_downstream_message(self, headers: Dict, message) -> None:
        print(headers)
        print(message)

    def on_upstream_message(self, headers: Dict, message) -> None:
        print(headers)
        print(message)


class SampleSecondaryAreaAgent(SecondaryAreaAgent):
    def __init__(
        self,
        upstream: MessageBusDefinition,
        downstream: MessageBusDefinition,
        config: Dict,
        feeder: Dict = None,
        simulation_id: str = None,
    ) -> None:
        super().__init__(upstream, downstream, config, feeder, simulation_id)
        self._latch = False
        qy.init_cim(self.secondary_area)
        self.branch_info, self.bus_info = qy.query_line_info(
            self.secondary_area)

        branch, bus = qy.query_transformers(self.secondary_area)
        self.branch_info.update(branch)
        self.bus_info.update(bus)

        self.bus_info.update(qy.query_power_electronics(self.secondary_area))
        self.bus_info.update(qy.query_energy_consumers(self.secondary_area))

        print(f'branch count: {len(self.branch_info.keys())}')
        print(f'bus count:  {len(self.bus_info.keys())}')

        self.connect()

    def on_measurement(self, headers: Dict, message):
        if not self._latch:
            log.debug(
                "measurement: %s.%s",
                self.__class__.__name__,
                headers.get("destination"),
                exc_info=True,
            )
            with open(f"{os.environ.get('OUTPUT_DIR')}/secondary.json", "w", encoding="UTF-8") as file:
                file.write(json.dumps(message))
            self._latch = True


def overwrite_parameters(feeder_id: str, area_id: str = "") -> MessageBusDefinition:
    bus_def = MessageBusDefinition.load(os.environ.get("BUS_CONFIG"))
    if area_id:
        bus_def.id = feeder_id + "." + area_id
    else:
        bus_def.id = feeder_id

    address = os.environ.get("GRIDAPPSD_ADDRESS")
    port = os.environ.get("GRIDAPPSD_PORT")
    if not address or not port:
        raise ValueError(
            "import auth_context or set environment up before this statement."
        )

    bus_def.conneciton_args["GRIDAPPSD_ADDRESS"] = f"tcp://{address}:{port}"
    bus_def.conneciton_args["GRIDAPPSD_USER"] = os.environ.get(
        "GRIDAPPSD_USER")
    bus_def.conneciton_args["GRIDAPPSD_PASSWORD"] = os.environ.get(
        "GRIDAPPSD_PASSWORD")
    print(bus_def)
    return bus_def


def save_area(context: dict) -> None:
    with open(
        f"{os.environ.get('OUTPUT_DIR')}/{context['message_bus_id']}.json", "w", encoding="UTF-8"
    ) as file:
        file.write(json.dumps(context))


def save_info(context: str, info: dict) -> None:
    with open(f"{os.environ.get('OUTPUT_DIR')}/{context}.json", "w", encoding="UTF-8") as file:
        file.write(json.dumps(info))


def spawn_feeder_agent(
        context: dict, sim: Sim, coord_agent: SampleCoordinatingAgent):
    save_area(context)

    agent_config = {
        "app_id": "sample_app",
        "description":
        "This is a GridAPPS-D sample distribution application agent"
    }

    agent = SampleFeederAgent(
        context, context, agent_config, None, sim.get_simulation_id())
    coord_agent.spawn_distributed_agent(agent)

    return agent


def spawn_secondary_agents(
    context: dict,
    sw_idx: int,
    sec_idx: int,
    sim: Sim,
    coord_agent: SampleCoordinatingAgent,
) -> None:
    switch_message_bus = overwrite_parameters(sim.get_feeder_id(), f"{sw_idx}")
    secondary_area_message_bus_def = overwrite_parameters(
        sim.get_feeder_id(), f"{sw_idx}.{sec_idx}"
    )
    save_area(context)

    agent_config = {
        "app_id": "something_else",
        "description":
        "This is a GridAPPS-D sample distribution application agent"
    }

    agent = SampleSecondaryAreaAgent(
        switch_message_bus,
        secondary_area_message_bus_def,
        agent_config,
        context,
        sim.get_simulation_id(),
    )
    if len(agent.secondary_area.addressable_equipment) > 1:
        coord_agent.spawn_distributed_agent(agent)


def spawn_switch_area_agents(
    context: dict, idx: int, sim: Sim, coord_agent: SampleCoordinatingAgent
) -> None:
    feeder_message_bus = overwrite_parameters(sim.get_feeder_id())
    save_area(context)

    agent_config = {
        "app_id": "something_new",
        "description":
        "This is a GridAPPS-D sample distribution application agent"
    }

    switch_message_bus = overwrite_parameters(sim.get_feeder_id(), f"{idx}")
    agent = SampleSwitchAreaAgent(
        feeder_message_bus, switch_message_bus, agent_config, context, sim.get_simulation_id()
    )
    coord_agent.spawn_distributed_agent(agent)
