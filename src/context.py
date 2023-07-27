
import logging
from typing import Dict

import gridappsd.topics as t
from gridappsd import GridAPPSD
from gridappsd.field_interface.agents import (FeederAgent, SecondaryAreaAgent,
                                              SwitchAreaAgent)
from gridappsd.field_interface.interfaces import MessageBusDefinition

log = logging.getLogger(__name__)

REQUEST_FIELD = ".".join((t.PROCESS_PREFIX, "request.field"))
REQUEST_FIELD_CONTEXT = ".".join((REQUEST_FIELD, "context"))


class FeederAreaContextManager(FeederAgent):

    def __init__(self,
                 upstream: MessageBusDefinition,
                 downstream: MessageBusDefinition,
                 config: Dict,
                 area: Dict = None,
                 simulation_id: str = None) -> None:

        self.ot_connection = GridAPPSD()
        if area is None:
            request = {'modelId': downstream.id}
            area = self.ot_connection.get_response(
                REQUEST_FIELD_CONTEXT, request, timeout=10)['data']

        super().__init__(upstream, downstream, config, area, simulation_id)

        # Override agent_id to a static value
        self.agent_id = downstream.id + '.context_manager'

        self.context = None

        self.registered_agents = {}
        self.registered_agents[self.agent_id] = self.get_registration_details()

        self.neighbouring_agents = {}
        self.upstream_agents = {}
        self.downstream_agents = {}

    def on_request(self, message_bus, headers: Dict, message):

        log.debug(f"Received request: {message}")

        if message['request_type'] == 'get_context':
            del message['request_type']
            reply_to = headers['reply-to']
            if self.context is None:
                self.context = self.ot_connection.get_response(
                    REQUEST_FIELD_CONTEXT, message)
            message_bus.send(reply_to, self.context)

        elif message['request_type'] == 'register_agent':
            self.ot_connection.send(t.REGISTER_AGENT_QUEUE, message)
            self.registered_agents[message['agent']
                                   ['agent_id']] = message['agent']

        elif message['request_type'] == 'get_agents':
            reply_to = headers['reply-to']
            message_bus.send(reply_to, self.registered_agents)


class SwitchAreaContextManager(SwitchAreaAgent):

    def __init__(self,
                 upstream: MessageBusDefinition,
                 downstream: MessageBusDefinition,
                 config: Dict,
                 area: Dict = None,
                 simulation_id: str = None) -> None:

        self.ot_connection = GridAPPSD()
        if area is None:
            request = {'areaId': downstream.id}
            area = self.ot_connection.get_response(
                REQUEST_FIELD_CONTEXT, request, timeout=10)['data']

        super().__init__(upstream, downstream, config, area, simulation_id)

        # Override agent_id to a static value
        self.agent_id = downstream.id + '.context_manager'

        self.context = None

        self.registered_agents = {}
        self.registered_agents[self.agent_id] = self.get_registration_details()

    def on_request(self, message_bus, headers: Dict, message):

        log.debug(f"Received request: {message}")

        if message['request_type'] == 'get_context':
            reply_to = headers['reply-to']
            if self.context is None:
                self.context = self.ot_connection.get_response(
                    REQUEST_FIELD_CONTEXT, message)
            message_bus.send(reply_to, self.context)

        elif message['request_type'] == 'register_agent':
            self.ot_connection.send(t.REGISTER_AGENT_QUEUE, message)
            self.registered_agents[message['agent']
                                   ['agent_id']] = message['agent']

        elif message['request_type'] == 'get_agents':
            reply_to = headers['reply-to']
            message_bus.send(reply_to, self.registered_agents)


class SecondaryAreaContextManager(SecondaryAreaAgent):

    def __init__(self,
                 upstream: MessageBusDefinition,
                 downstream: MessageBusDefinition,
                 config: Dict,
                 area: Dict = None,
                 simulation_id: str = None) -> None:

        self.ot_connection = GridAPPSD()
        if area is None:
            request = {'areaId': downstream.id}
            area = self.ot_connection.get_response(
                REQUEST_FIELD_CONTEXT, request, timeout=10)['data']

        super().__init__(upstream, downstream, config, area, simulation_id)

        # Override agent_id to a static value
        self.agent_id = downstream.id + '.context_manager'

        self.context = None

        self.registered_agents = {}
        self.registered_agents[self.agent_id] = self.get_registration_details()

    def on_request(self, message_bus, headers: Dict, message):

        log.debug(f"Received request: {message}")
        log.debug(f"Received request: {headers}")

        if message['request_type'] == 'get_context':
            reply_to = headers['reply-to']
            if self.context is None:
                self.context = self.ot_connection.get_response(
                    REQUEST_FIELD_CONTEXT, message)
            message_bus.send(reply_to, self.context)

        elif message['request_type'] == 'register_agent':
            self.ot_connection.send(t.REGISTER_AGENT_QUEUE, message)
            self.registered_agents[message['agent']
                                   ['agent_id']] = message['agent']

        elif message['request_type'] == 'get_agents':
            reply_to = headers['reply-to']
            message_bus.send(reply_to, self.registered_agents)
