import os
import traceback
from typing import Dict
from typing import OrderedDict
from typing import Tuple
import json
import logging
import numpy as np
import math
import gridappsd.topics as t
from gridappsd import DifferenceBuilder
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
NEIGHBOR_BUSES = {"18": "135", "135": "18", "13": "152",
                  "152": "13", "60": "160", "160": "60", "97": "197", "197": "97"}
SOURCE_VOLTAGE = [1.0175, 1.0175, 1.0175]
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


def pol_to_cart(rho: float, phi: float) -> Tuple[float, float]:
    rad = math.radians(phi)
    x = rho * np.cos(rad)
    y = rho * np.sin(rad)
    return x, y


class SampleCoordinatingAgent(CoordinatingAgent):
    def __init__(self, system: MessageBusDefinition, simulation_id=None):
        super().__init__(None, system, simulation_id)
        log.debug("Spawning Coordinating Agent")


class SampleFeederAgent(FeederAgent):
    def __init__(self,
                 upstream: MessageBusDefinition,
                 downstream: MessageBusDefinition,
                 config: Dict,
                 area: Dict = None,
                 simulation_id: str = None) -> None:
        self._latch = False
        super().__init__(upstream, downstream, config, area, simulation_id)
        log.debug("Spawning Feeder Agent")

    def on_measurement(self, headers: Dict, message) -> None:
        if not self._latch:
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
    def __init__(self,
                 upstream: MessageBusDefinition,
                 downstream: MessageBusDefinition,
                 config: Dict,
                 area: Dict = None,
                 simulation_id: str = None) -> None:
        self.counter = 0
        self.last_update = 0
        self._location = ""
        self.source_received = False
        self.bus_info = {}
        self.branch_info = {}
        self.mrid_map = {}
        super().__init__(upstream, downstream, config, area, simulation_id)
        self.alpha = AlphaArea()
        qy.init_cim(self.switch_area)
        qy.query_line_info(self.switch_area, self.bus_info, self.branch_info)

        qy.query_transformers(
            self.switch_area, self.bus_info, self.branch_info)

        qy.query_power_electronics(
            self.switch_area, self.bus_info, self.mrid_map)

        qy.query_energy_consumers(
            self.switch_area, self.bus_info, self.mrid_map)

        qy.index_info(self.branch_info, self.bus_info)

        for key, value in SOURCE_BUSES.items():
            if value in self.bus_info:
                self._location = key

        if self._location == "":
            raise Exception("Area has no source bus!")

        self.source_bus = SOURCE_BUSES[self._location]
        self.source_line = qy.source_line(self.branch_info, self.source_bus)

        log.debug(f'branch count: {len(self.branch_info.keys())}')
        log.debug(f'bus count:  {len(self.bus_info.keys())}')
        log.debug(f'source bus:  {self.source_bus}')
        log.debug(f'source line:  {self.source_line}')

        save_info(
            f'{area["message_bus_id"]}_branch_info_{self._location}', OrderedDict(
                sorted(self.branch_info.items())),
        )
        save_info(
            f'{area["message_bus_id"]}_businfo_{self._location}',
            OrderedDict(sorted(self.bus_info.items())),
        )
        save_info(
            f'{area["message_bus_id"]}_mrids_{self._location}', OrderedDict(
                sorted(self.mrid_map.items())),
        )

    def on_measurement(self, headers: Dict, message):
        if self._location == '':
            return None

        for key, value in message.items():
            if key in self.mrid_map:
                bus = self.mrid_map[key]

                if key in self.bus_info[bus]['mrid']:
                    real, imag = pol_to_cart(
                        value['magnitude'], value['angle'])
                    mrid_idx = self.bus_info[bus]['mrid'].index(key)
                    type = self.bus_info[bus]['types'][mrid_idx]
                    phase = self.bus_info[bus]['phases'][mrid_idx]
                    self.bus_info[bus][type][phase-1] = [real, imag]

        time = int(headers['timestamp'])
        if time % 60 == 0 and time != self.last_update:
            self.counter = 0
            self.last_update = time
            self.admm()

    def on_upstream_message(self, headers: Dict, message) -> None:
        if self._location == '':
            return None

        if message['bus'] in self.bus_info:
            log.debug(
                f"Area {self._location} received message from upstream message bus: {message}")
            self.bus_info[message['bus']]['pq'] = message['pq']
            self.admm()

    def on_downstream_message(self, headers: Dict, message) -> None:
        log.debug(f"Received message from downstream message bus: {message}")

    def admm(self):
        if self.counter > 5:
            return None

        try:
            [voltages, flows, alpha, pi, qi] = self.alpha.alpha_area(
                self.branch_info,
                self.bus_info,
                self.source_bus,
                self.bus_info[self.source_bus]["idx"],
                SOURCE_VOLTAGE,
                0,
                False,
                int(self._location),
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

            if self.source_bus in NEIGHBOR_BUSES:
                neighbor_bus = NEIGHBOR_BUSES[self.source_bus]

                message = {
                    "bus": neighbor_bus,
                    "pq": list(flows[self.source_line].values()),
                    "alpha": alpha
                }
                self.publish_upstream(message)

                if self.counter == 5:
                    for bus, info in self.bus_info.items():
                        for i, type in enumerate(info['types']):
                            if type == 'pv':
                                device_id = info['control_mrid'][i]
                                attribute = "PowerElectronicsConnection.p"
                                phase = info['phases'][i]-1
                                [old_real, old_imag] = info[type][phase]
                                new_real = old_real*(1.0 - alpha)
                                log.debug(
                                    f"Area {self._location} Control : {device_id}, {old_real} -> {new_real}")
                                difference_builder = DifferenceBuilder(
                                    self.simulation_id)
                                difference_builder.add_difference(
                                    device_id, attribute, new_real, old_real)
                                self.send_control_command(difference_builder)

            self.counter += 1
        except Exception as e:
            log.debug(f"Area {self._location}")
            log.debug(e)
            log.debug(traceback.format_exc())
            save_info(
                f'debug_branch_info_{self._location}', OrderedDict(
                    sorted(self.branch_info.items())),
            )
            save_info(
                f'debug_businfo_{self._location}',
                OrderedDict(sorted(self.bus_info.items())),
            )


class SampleSecondaryAreaAgent(SecondaryAreaAgent):
    def __init__(self,
                 upstream: MessageBusDefinition,
                 downstream: MessageBusDefinition,
                 config: Dict,
                 area: Dict = None,
                 simulation_id: str = None) -> None:
        self._location = ""
        self._measurements = {}
        self.bus_info = {}
        self.branch_info = {}
        self.mrid_map = {}
        super().__init__(upstream, downstream, config, area, simulation_id)
        self.alpha = AlphaArea()
        qy.init_cim(self.switch_area)
        qy.query_line_info(self.switch_area, self.bus_info, self.branch_info)

        qy.query_transformers(
            self.switch_area, self.bus_info, self.branch_info)

        qy.query_power_electronics(
            self.switch_area, self.bus_info, self.mrid_map)

        qy.query_energy_consumers(
            self.switch_area, self.bus_info, self.mrid_map)

        qy.index_info(self.branch_info, self.bus_info)

        for key, value in SOURCE_BUSES.items():
            if value in self.bus_info:
                self._location = key
            else:
                raise Exception("Area has no source bus!")

        log.debug(f'branch count: {len(self.branch_info.keys())}')
        log.debug(f'bus count:  {len(self.bus_info.keys())}')
        log.debug(f'source bus:  {SOURCE_BUSES[str(self._location)]}')

        save_info(
            f'{area["message_bus_id"]}_branch_info_{self._location}', OrderedDict(
                sorted(self.branch_info.items())),
        )
        save_info(
            f'{area["message_bus_id"]}_businfo_{self._location}',
            OrderedDict(sorted(self.bus_info.items())),
        )
        save_info(
            f'{area["message_bus_id"]}_mrids_{self._location}', OrderedDict(
                sorted(self.mrid_map.items())),
        )

    def on_measurement(self, headers: Dict, message):
        for key, value in message.items():
            if key in self.mrid_map:
                real, imag = pol_to_cart(value['magnitude'], value['angle'])
            if key in self._measurements:
                with open(f"{os.environ.get('OUTPUT_DIR')}/measurments_{self._location}.json", "w", encoding="UTF-8") as file:
                    file.write(json.dumps(self._measurements))
                self._measurements = {}
            else:
                self._measurements[key] = value

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
