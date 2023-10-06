import os
import logging
from typing import Tuple
from enum import IntEnum
import numpy as np

import gridappsd.field_interface.agents.agents as agents_mod
from cimgraph.data_profile import CIM_PROFILE
cim_profile = CIM_PROFILE.RC4_2021.value
agents_mod.set_cim_profile(cim_profile)
cim = agents_mod.cim

log = logging.getLogger(__name__)

OUTPUT_DIR = os.environ.get("OUTPUTS")


class Phase(IntEnum):
    A = 1
    B = 2
    C = 3

    def __repr__(self):
        return self.value


def source_line(branch_info: dict, bus: str) -> str:
    for key, value in branch_info.items():
        if value['fr_bus'] == bus or value['to_bus'] == bus:
            return key

    raise Exception(f"Source line not found for source bus: {bus}!")


def init_branch() -> dict:
    branch = {}
    branch["phases"] = []
    branch["zprim"] = np.zeros((3, 3, 2)).tolist()
    return branch


def init_bus() -> dict:
    bus = {}
    bus["phases"] = []
    bus["types"] = []
    bus["mrid"] = []
    bus["kv"] = 2400  # TODO query actual values
    bus["pq"] = np.zeros((3, 2)).tolist()
    bus["pv"] = np.zeros((3, 2)).tolist()
    return bus


def index_info(branch: dict, bus: dict):
    for i, name in enumerate(bus):
        bus[name]["idx"] = i

    for i, name in enumerate(branch):
        branch[name]["idx"] = i
        branch[name]["from"] = bus[branch[name]["fr_bus"]]["idx"]
        branch[name]["to"] = bus[branch[name]["to_bus"]]["idx"]


def init_cim(network_area) -> None:
    network_area.get_all_attributes(cim.ACLineSegment)
    network_area.get_all_attributes(cim.ACLineSegmentPhase)
    network_area.get_all_attributes(cim.BaseVoltage)
    network_area.get_all_attributes(cim.SvEstVoltage)
    network_area.get_all_attributes(cim.SvVoltage)
    network_area.get_all_attributes(cim.Equipment)
    network_area.get_all_attributes(cim.PerLengthPhaseImpedance)
    network_area.get_all_attributes(cim.PhaseImpedanceData)
    network_area.get_all_attributes(cim.WireSpacingInfo)
    network_area.get_all_attributes(cim.WirePosition)
    network_area.get_all_attributes(cim.OverheadWireInfo)
    network_area.get_all_attributes(cim.ConcentricNeutralCableInfo)
    network_area.get_all_attributes(cim.TapeShieldCableInfo)
    network_area.get_all_attributes(cim.TransformerTank)
    network_area.get_all_attributes(cim.TransformerTankEnd)
    network_area.get_all_attributes(cim.TransformerTankInfo)
    network_area.get_all_attributes(cim.TransformerEndInfo)
    network_area.get_all_attributes(cim.ShortCircuitTest)
    network_area.get_all_attributes(cim.NoLoadTest)
    network_area.get_all_attributes(cim.PowerElectronicsConnection)
    network_area.get_all_attributes(cim.PowerElectronicsConnectionPhase)
    network_area.get_all_attributes(cim.EnergyConsumer)
    network_area.get_all_attributes(cim.EnergyConsumerPhase)
    network_area.get_all_attributes(cim.Analog)
    network_area.get_all_attributes(cim.EnergyConsumerPhase)
    network_area.get_all_attributes(cim.Terminal)


def query_line_info(network_area, bus_info: dict, branch_info: dict):
    if cim.ACLineSegment not in network_area.typed_catalog:
        return None

    line_ids = list(network_area.typed_catalog[cim.ACLineSegment].keys())

    for line_id in line_ids:
        line = network_area.typed_catalog[cim.ACLineSegment][line_id]
        branch_info[line.name] = init_branch()
        branch_info[line.name]["type"] = "LINE"

        from_bus = line.Terminals[0].ConnectivityNode.name
        branch_info[line.name]["fr_bus"] = from_bus

        if from_bus not in bus_info:
            bus_info[from_bus] = init_bus()

        to_bus = line.Terminals[1].ConnectivityNode.name
        branch_info[line.name]["to_bus"] = to_bus

        if to_bus not in bus_info:
            bus_info[to_bus] = init_bus()

        if line.PerLengthImpedance.PhaseImpedanceData is not None:
            length = float(line.length)
            for data in line.PerLengthImpedance.PhaseImpedanceData:
                row = int(data.row) - 1
                col = int(data.column) - 1
                real = length * float(data.r)
                imag = length * float(data.x)
                branch_info[line.name]["zprim"][row][col] = [real, imag]
                branch_info[line.name]["zprim"][col][row] = [real, imag]

        for line_phs in line.ACLineSegmentPhases:
            if not branch_info[line.name]["phases"]:
                branch_info[line.name]["phases"].append(
                    Phase[line_phs.phase[0]])


def query_transformers(network_area, bus_info: dict, branch_info: dict):
    if cim.TransformerTank not in network_area.typed_catalog:
        return None

    for tank in network_area.typed_catalog[cim.TransformerTank].values():
        name = tank.TransformerTankEnds[0].Terminal.name[:-3]
        phase = Phase[name[-1:].upper()]

        if phase == Phase.A:
            if name not in branch_info:
                branch_info[name] = init_branch()

            branch_info[name]["type"] = "XFMR"
            branch_info[name]["phases"] = [
                1,
                2,
                3,
            ]  # TODO this should be captured during iteration, but its a pain

            from_node = tank.TransformerTankEnds[0].Terminal.ConnectivityNode
            branch_info[name]["fr_bus"] = from_node.name

            to_node = tank.TransformerTankEnds[1].Terminal.ConnectivityNode
            branch_info[name]["to_bus"] = to_node.name

            for end_info in tank.TransformerTankInfo.TransformerEndInfos:

                if from_node.name not in bus_info:
                    bus_info[from_node.name] = init_bus()

                if to_node.name not in bus_info:
                    bus_info[to_node.name] = init_bus()

                bus_info[from_node.name]["kv"] = float(end_info.ratedU)
                bus_info[to_node.name]["kv"] = float(end_info.ratedU)


def query_power_electronics(network_area, bus_info: dict, mrid_map: dict):
    if cim.PowerElectronicsConnection not in network_area.typed_catalog:
        return None

    for pec in network_area.typed_catalog[cim.PowerElectronicsConnection].values():
        node = pec.Terminals[0].ConnectivityNode

        if node.name not in bus_info:
            bus_info[node.name] = init_bus()

        bus_info[node.name]["kv"] = float(pec.ratedU)

        for pec_phs in pec.PowerElectronicsConnectionPhases:
            mrid = pec_phs.PowerElectronicsConnection.Measurements[0].mRID
            mrid_map[mrid] = node.name

            log.debug(
                f"{node.name}:{mrid} p={pec.p}, q={pec.q} on phase={pec_phs.phase}")
            log.debug(f"{node.name}: {mrid} pec={pec.mRID}")

            if pec_phs.phase:
                power = [float(pec_phs.p), float(pec_phs.q)]
                if pec_phs.phase[0] == "A":
                    bus_info[node.name]["mrid"].append(mrid)
                    bus_info[node.name]["pv"][0] = power
                    bus_info[node.name]["phases"].append(
                        Phase[pec_phs.phase[0]])
                    bus_info[node.name]["types"].append("pv")

                if pec_phs.phase[0] == "B":
                    bus_info[node.name]["mrid"].append(mrid)
                    bus_info[node.name]["pv"][1] = power
                    bus_info[node.name]["phases"].append(
                        Phase[pec_phs.phase[0]])
                    bus_info[node.name]["types"].append("pv")

                if pec_phs.phase[0] == "C":
                    bus_info[node.name]["mrid"].append(mrid)
                    bus_info[node.name]["pv"][2] = power
                    bus_info[node.name]["phases"].append(
                        Phase[pec_phs.phase[0]])
                    bus_info[node.name]["types"].append("pv")
            else:
                real = float(pec.p)/3.0
                imag = float(pec.q)/3.0
                bus_info[node.name]["mrid"].append(mrid)
                bus_info[node.name]["pv"][0] = [real, imag]
                bus_info[node.name]["pv"][1] = [real, imag]
                bus_info[node.name]["pv"][2] = [real, imag]
                bus_info[node.name]["phases"].append('')


def query_energy_consumers(network_area, bus_info: dict, mrid_map: dict):
    if cim.EnergyConsumer not in network_area.typed_catalog:
        return None

    for load in network_area.typed_catalog[cim.EnergyConsumer].values():
        node = load.Terminals[0].ConnectivityNode

        if node.name not in bus_info:
            bus_info[node.name] = init_bus()

        for load_phs in load.EnergyConsumerPhase:
            if load_phs.phase:
                power = [float(load_phs.p), float(load_phs.q)]
                mrid = load.Measurements[0].mRID
                mrid_map[mrid] = node.name

                if load_phs.phase[0] == "A":
                    bus_info[node.name]["mrid"].append(mrid)
                    bus_info[node.name]["pq"][0] = power
                    bus_info[node.name]["phases"].append(
                        Phase[load_phs.phase[0]])
                    bus_info[node.name]["types"].append("pq")

                if load_phs.phase[0] == "B":
                    bus_info[node.name]["mrid"].append(mrid)
                    bus_info[node.name]["pq"][1] = power
                    bus_info[node.name]["phases"].append(
                        Phase[load_phs.phase[0]])
                    bus_info[node.name]["types"].append("pq")

                if load_phs.phase[0] == "C":
                    bus_info[node.name]["mrid"].append(mrid)
                    bus_info[node.name]["pq"][2] = power
                    bus_info[node.name]["phases"].append(
                        Phase[load_phs.phase[0]])
                    bus_info[node.name]["types"].append("pq")
            else:
                if load_phs.p is not None:
                    real = float(load_phs.p)/3.0
                    imag = float(load_phs.q)/3.0
                    bus_info[node.name]["mrid"].append(mrid)
                    bus_info[node.name]["pq"][0] = [real, imag]
                    bus_info[node.name]["pq"][1] = [real, imag]
                    bus_info[node.name]["pq"][2] = [real, imag]
                    bus_info[node.name]["phases"].append('')
