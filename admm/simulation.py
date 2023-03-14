import json
import logging
from gridappsd import GridAPPSD
from gridappsd.simulation import Simulation
from gridappsd import topics as t

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

class Sim(object):
    """
        https://gridappsd-training.readthedocs.io/en/develop/api_usage/3.6-Controlling-Simulation-API.html
    """
    done = False

    def __init__(self, config: dict) -> None:
        """_summary_
        """
        self.gapps = GridAPPSD()
        assert self.gapps.connected
        
        self.config = config
        self.simulation = Simulation(self.gapps, self.config)
        self.simulation.add_oncomplete_callback(self.on_complete)
        self.simulation.add_onmeasurement_callback(self.on_measurment)
        self.simulation.add_onstart_callback(self.on_start)
        self.simulation.add_ontimestep_callback(self.on_timestep)
        self.simulation.start_simulation()
        
        self.gapps.subscribe(t.simulation_output_topic(self.get_simulation_id()), self.on_message)
        # self.diff = DifferenceBuilder(simulation_id)
        
    def get_feeder_id(self) -> str:
        """_summary_
        """
        return self.config["power_system_config"]["Line_name"]
    
    def get_simulation_id(self) -> str:
        """_summary_
        """
        return self.simulation.simulation_id
    
    def on_start(self, sim) -> None:
        """_summary_
        """
        # Use extra methods to subscribe to other topics, such as simulation logs
        print(f"The simulation has started with id : {sim.simulation_id}")

    def on_measurment(self, sim, timestamp, measurements) -> None:
        """_summary_
        """
        # tm = timestamp
        # print(f"A measurement was taken with timestamp : {timestamp}")
        # Print the switch status just once
        # switch_data = self._switch_df[self._switch_df['eqid'] == '_6C1FDA90-1F4E-4716-BC90-1CCB59A6D5A9']
        # print(switch_data)
        # for k in switch_data.index:
        #     measid = switch_data['measid'][k]
        #     status = measurements[measid]['value']
        #     (switch_data, status)

    def on_timestep(self, sim, timestep) -> None:
        """_summary_
        """

    def on_complete(self, sim) -> None:
        """_summary_
        """
        print("The simulation has finished")
        self.done = True

    def on_message(self, headers, message) -> None:
        """_summary_
        """
        if isinstance(message,str):
            message = json.loads(message)
        else:
            pass

        if 'message' not in message:
            if message['processStatus'] == ('COMPLETE' or 'CLOSED'):
                print('End of Simulation')
                self.done = True
                