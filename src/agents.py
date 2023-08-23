import os
from typing import Dict
from typing import OrderedDict
import json
import logging
import numpy as np
import gridappsd.topics as t
from gridappsd.field_interface.context import LocalContext
from gridappsd.field_interface.interfaces import MessageBusDefinition
from gridappsd.field_interface.agents import CoordinatingAgent
from gridappsd.field_interface.agents import FeederAgent
from gridappsd.field_interface.agents import SwitchAreaAgent
from gridappsd.field_interface.agents import SecondaryAreaAgent
import queries as qy
from alpha_area import AlphaArea

log = logging.getLogger(__name__)

REQUEST_FIELD = ".".join((t.PROCESS_PREFIX, "request.field"))
REQUEST_FIELD_CONTEXT = ".".join((REQUEST_FIELD, "context"))

SOURCE_BUSES = {"1": "149", "2": "135", "3": "152", "4": "160", "5": "197"}
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
    _latch = False

    def __init__(self, system: MessageBusDefinition, simulation_id=None):
        super().__init__(None, system, simulation_id)
        log.debug("Spawning Coordinating Agent")


class SampleFeederAgent(FeederAgent):
    _latch = False

    def __init__(self,
                 upstream: MessageBusDefinition,
                 downstream: MessageBusDefinition,
                 config: Dict,
                 area: Dict = None,
                 simulation_id: str = None) -> None:
        super().__init__(upstream, downstream, config, area, simulation_id)
        log.debug("Spawning Feeder Agent")

    def on_measurement(self, headers: Dict, message) -> None:
        if not self._latch:
            log.debug(f"feeder on measurment message: {message}")
            with open(f"{os.environ.get('OUTPUT_DIR')}/feeder.json", "a", encoding="UTF-8") as file:
                file.write(json.dumps(message))
            self._latch = True

    def on_upstream_message(self, headers: Dict, message) -> None:
        log.debug(f"Received message from upstream message bus: {message}")

    def on_downstream_message(self, headers: Dict, message) -> None:
        log.debug(f"Received message from downstream message bus: {message}")

    def on_request(self, message_bus, headers: Dict, message):
        print(f"Received request: {message}")

        reply_to = headers['reply-to']

        message_bus.send(reply_to, 'this is a reponse')


class SampleSwitchAreaAgent(SwitchAreaAgent):
    _latch = False
    _location = ""

    def __init__(self,
                 upstream: MessageBusDefinition,
                 downstream: MessageBusDefinition,
                 config: Dict,
                 area: Dict = None,
                 simulation_id: str = None) -> None:
        self._measurements = {}
        super().__init__(upstream, downstream, config, area, simulation_id)
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

        log.debug(f'branch count: {len(self.branch_info.keys())}')
        log.debug(f'bus count:  {len(self.bus_info.keys())}')

        for key, value in SOURCE_BUSES.items():
            if value in self.bus_info:
                self._location = key

        save_info(
            f'{area["message_bus_id"]}_branch_info_{self._location}', OrderedDict(
                sorted(self.branch_info.items())),
        )
        save_info(
            f'{area["message_bus_id"]}_businfo_{self._location}',
            OrderedDict(sorted(self.bus_info.items())),
        )

    def on_measurement(self, headers: Dict, message):
        if not self._latch:
            for key, value in message.items():
                if key in self._measurements:
                    with open(f"{os.environ.get('OUTPUT_DIR')}/measurments_{self._location}.json", "w", encoding="UTF-8") as file:
                        file.write(json.dumps(self._measurements))
                    self._latch
                else:
                    self._measurements[key] = value
            # self._latch = True

    def on_upstream_message(self, headers: Dict, message) -> None:
        log.info(f"Received message from upstream message bus: {message}")

    def on_downstream_message(self, headers: Dict, message) -> None:
        log.info(f"Received message from downstream message bus: {message}")


class SampleSecondaryAreaAgent(SecondaryAreaAgent):
    _latch = False

    def __init__(self,
                 upstream: MessageBusDefinition,
                 downstream: MessageBusDefinition,
                 config: Dict,
                 area: Dict = None,
                 simulation_id: str = None) -> None:
        super().__init__(upstream, downstream, config, area, simulation_id)
        qy.init_cim(self.secondary_area)
        self.branch_info, self.bus_info = qy.query_line_info(
            self.secondary_area)

        branch, bus = qy.query_transformers(self.secondary_area)
        self.branch_info.update(branch)
        self.bus_info.update(bus)

        self.bus_info.update(qy.query_power_electronics(self.secondary_area))
        self.bus_info.update(qy.query_energy_consumers(self.secondary_area))

        log.debug(f'branch count: {len(self.branch_info.keys())}')
        log.debug(f'bus count:  {len(self.bus_info.keys())}')

    def on_measurement(self, headers: Dict, message):
        if not self._latch:
            with open(f"{os.environ.get('OUTPUT_DIR')}/secondary.json", "w", encoding="UTF-8") as file:
                file.write(json.dumps(message))
            self._latch = True

    def on_upstream_message(self, headers: Dict, message) -> None:
        log.info(f"Received message from upstream message bus: {message}")

    def on_downstream_message(self, headers: Dict, message) -> None:
        log.info(f"Received message from downstream message bus: {message}")


def overwrite_parameters(feeder_id: str, area_id: str = "") -> MessageBusDefinition:
    bus = MessageBusDefinition.load(os.environ.get("BUS_CONFIG"))
    if area_id:
        bus.id = feeder_id + "." + area_id
    else:
        bus.id = feeder_id

    address = os.environ.get("GRIDAPPSD_ADDRESS")
    port = os.environ.get("GRIDAPPSD_PORT")
    if not address or not port:
        raise ValueError(
            "import auth_context or set environment up before this statement."
        )

    bus.conneciton_args["GRIDAPPSD_ADDRESS"] = f"tcp://{address}:{port}"
    bus.conneciton_args["GRIDAPPSD_USER"] = os.environ.get(
        "GRIDAPPSD_USER")
    bus.conneciton_args["GRIDAPPSD_PASSWORD"] = os.environ.get(
        "GRIDAPPSD_PASSWORD")
    return bus


def save_area(context: dict) -> None:
    with open(
        f"{os.environ.get('OUTPUT_DIR')}/{context['message_bus_id']}.json", "w", encoding="UTF-8"
    ) as file:
        file.write(json.dumps(context))


def save_info(context: str, info: dict) -> None:
    with open(f"{os.environ.get('OUTPUT_DIR')}/{context}.json", "w", encoding="UTF-8") as file:
        file.write(json.dumps(info))
