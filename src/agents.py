import os
from typing import Dict
import json
import logging
import numpy as np
from gridappsd.field_interface.context import LocalContext
from gridappsd.field_interface.interfaces import MessageBusDefinition
from gridappsd.field_interface.agents import CoordinatingAgent
from gridappsd.field_interface.agents import FeederAgent
from gridappsd.field_interface.agents import SwitchAreaAgent
from gridappsd.field_interface.agents import SecondaryAreaAgent
import queries as qy

log = logging.getLogger(__name__)


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
    def __init__(self, system: MessageBusDefinition, simulation_id=None):
        super().__init__(None, system, simulation_id)
        log.debug("Spawning Coordinating Agent")


class SampleFeederAgent(FeederAgent):
    def __init__(self,
                 upstream: MessageBusDefinition,
                 downstream: MessageBusDefinition,
                 config: Dict,
                 feeder_dict: Dict = None,
                 simulation_id: str = None) -> None:
        super().__init__(upstream, downstream, config, feeder_dict, simulation_id)
        log.debug("Spawning Feeder Agent")

    def on_measurement(self, headers: Dict, message) -> None:
        log.debug(
            f"measurement: {self.__class__.__name__}.{headers.get('destination')}"
        )
        with open("feeder.txt", "a") as fp:
            fp.write(json.dumps(message))

    def on_upstream_message(self, headers: Dict, message) -> None:
        log.debug(f"Received message from upstream message bus: {message}")

    def on_downstream_message(self, headers: Dict, message) -> None:
        log.debug(f"Received message from downstream message bus: {message}")

    def on_request(self, message_bus, headers: Dict, message):
        print(f"Received request: {message}")

        reply_to = headers['reply-to']

        message_bus.send(reply_to, 'this is a reponse')


class SampleSwitchAreaAgent(SwitchAreaAgent):
    def __init__(self,
                 upstream: MessageBusDefinition,
                 downstream: MessageBusDefinition,
                 config: Dict,
                 feeder_dict: Dict = None,
                 simulation_id: str = None) -> None:
        super().__init__(upstream, downstream, config, feeder_dict, simulation_id)
        print("Spawning Switch Agent")

    def on_measurement(self, headers: Dict, message):
        log.debug(
            f"measurement: {self.__class__.__name__}.{headers.get('destination')}"
        )
        with open("switch_area.txt", "a") as fp:
            fp.write(json.dumps(message))

    def on_upstream_message(self, headers: Dict, message) -> None:
        log.info(f"Received message from upstream message bus: {message}")

    def on_downstream_message(self, headers: Dict, message) -> None:
        log.info(f"Received message from downstream message bus: {message}")


class SampleSecondaryAreaAgent(SecondaryAreaAgent):
    def __init__(self,
                 upstream: MessageBusDefinition,
                 downstream: MessageBusDefinition,
                 config: Dict,
                 feeder_dict: Dict = None,
                 simulation_id: str = None) -> None:
        super().__init__(upstream, downstream, config, feeder_dict, simulation_id)
        print("Spawning Switch Agent")

    def on_measurement(self, headers: Dict, message):
        log.debug(
            f"measurement: {self.__class__.__name__}.{headers.get('destination')}"
        )
        with open("secondary.txt", "a") as fp:
            fp.write(json.dumps(message))

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
