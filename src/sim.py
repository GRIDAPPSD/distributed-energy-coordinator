import json
import logging
from gridappsd import GridAPPSD
from gridappsd.simulation import Simulation
from gridappsd import topics as t

log = logging.getLogger(__name__)


class Sim(object):
    """
    https://gridappsd-training.readthedocs.io/en/develop/api_usage/3.6-Controlling-Simulation-API.html
    """

    done = False

    def __init__(self, config: dict) -> None:
        self.gapps = GridAPPSD()
        assert self.gapps.connected

        self.config = config
        self.simulation = Simulation(self.gapps, self.config)
        self.simulation.add_oncomplete_callback(self.on_complete)
        self.simulation.add_onmeasurement_callback(self.on_measurment)
        self.simulation.add_onstart_callback(self.on_start)
        self.simulation.add_ontimestep_callback(self.on_timestep)
        self.simulation.start_simulation()

        self.gapps.subscribe(
            t.simulation_output_topic(
                self.get_simulation_id()), self.on_message
        )
        # self.diff = DifferenceBuilder(simulation_id)

    def get_feeder_id(self) -> str:
        return self.config["power_system_config"]["Line_name"]

    def get_simulation_id(self) -> str:
        return self.simulation.simulation_id

    def on_start(self, sim) -> None:
        # Use extra methods to subscribe to other topics
        log.debug(f"The simulation has started with id : {sim.simulation_id}")

    def on_measurment(self, sim, timestamp, measurements) -> None:
        pass

    def on_timestep(self, sim, timestep) -> None:
        pass

    def on_complete(self, sim) -> None:
        log.debug("The simulation has finished")
        self.done = True

    def on_message(self, headers, message) -> None:
        if isinstance(message, str):
            message = json.loads(message)
        else:
            pass

        if "message" not in message:
            if message["processStatus"] == ("COMPLETE" or "CLOSED"):
                log.debug("End of Simulation")
                self.done = True
