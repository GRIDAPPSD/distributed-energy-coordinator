from SPARQLWrapper import SPARQLWrapper2#, JSON
import sys
# constants.py is used for configuring blazegraph.
import json

#******************************************************************************
# URL for blazegraph
# Using the default blazegraph installation as a standalone
#blazegraph_url = "http://localhost:9999/blazegraph/namespace/kb/sparql"

# When running the platform in the docker, the blazegraph URL can be found in
# /gridappsd/conf/pnnl.goss.gridappsd.cfg. At the time of writing (04/24/18),
# there are two URLs. One for calling from inside the docker container, and one
# for calling from outside the docker container.

# URL from inside the docker container:
# blazegraph_url = "http://blazegraph:8080/bigdata/sparql"

# URL from outside the docker container:
blazegraph_url = "http://localhost:8889/bigdata/sparql"

#******************************************************************************
# Prefix for blazegraph queries.

# cim100 is used in InsertMeasurements.py. Notice the lack of "greater than" at the end.
cim100 = '<http://iec.ch/TC57/CIM100#'
# Prefix for all queries.
prefix = """PREFIX r: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX c: {cimURL}>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
""".format(cimURL=cim100)
#******************************************************************************
sparql = SPARQLWrapper2(blazegraph_url)
#******************************************************************************

################################################################################
###################### Fetching All loads from CIM file ########################
################################################################################
def get_all_load_info(feeder_mrid):
    qstr_distributed_load = prefix + """SELECT ?name ?bus ?basev ?p ?q ?conn ?cnt ?pz ?qz ?pi ?qi ?pp ?qp ?pe ?qe ?fdrid (group_concat(distinct ?phs;separator="") as ?phases)
        WHERE
          { 
            ?s r:type c:EnergyConsumer.
            # feeder selection options - if all commented out, query matches all feeders
             VALUES ?fdrid """  +'{"' + str(feeder_mrid) +'"}'+"""
             ?s c:Equipment.EquipmentContainer ?fdr.
             ?fdr c:IdentifiedObject.mRID ?fdrid.
             ?s c:IdentifiedObject.name ?name.
             ?s c:ConductingEquipment.BaseVoltage ?bv.
             ?bv c:BaseVoltage.nominalVoltage ?basev.
             ?s c:EnergyConsumer.customerCount ?cnt.
             ?s c:EnergyConsumer.p ?p.
             ?s c:EnergyConsumer.q ?q.
             ?s c:EnergyConsumer.phaseConnection ?connraw.
               bind(strafter(str(?connraw),"PhaseShuntConnectionKind.") as ?conn)
             ?s c:EnergyConsumer.LoadResponse ?lr.
             ?lr c:LoadResponseCharacteristic.pConstantImpedance ?pz.
             ?lr c:LoadResponseCharacteristic.qConstantImpedance ?qz.
             ?lr c:LoadResponseCharacteristic.pConstantCurrent ?pi.
             ?lr c:LoadResponseCharacteristic.qConstantCurrent ?qi.
             ?lr c:LoadResponseCharacteristic.pConstantPower ?pp.
             ?lr c:LoadResponseCharacteristic.qConstantPower ?qp.
             ?lr c:LoadResponseCharacteristic.pVoltageExponent ?pe.
             ?lr c:LoadResponseCharacteristic.qVoltageExponent ?qe.
             OPTIONAL {?ecp c:EnergyConsumerPhase.EnergyConsumer ?s.
             ?ecp c:EnergyConsumerPhase.phase ?phsraw.
               bind(strafter(str(?phsraw),"SinglePhaseKind.") as ?phs) }
             ?t c:Terminal.ConductingEquipment ?s.
             ?t c:Terminal.ConnectivityNode ?cn. 
             ?cn c:IdentifiedObject.name ?bus
            }
            GROUP BY ?name ?bus ?basev ?p ?q ?cnt ?conn ?pz ?qz ?pi ?qi ?pp ?qp ?pe ?qe ?fdrid
            ORDER by ?name
        """
    sparql.setQuery(qstr_distributed_load)
    ret = sparql.query()
    Property = {}
    for b in ret.bindings:
        for keys in b:
            if 'name' in keys:
                object_name = b[keys].value
                Property[object_name] = {}
            else:
                Property[object_name][keys] = b[keys].value

    return Property

################################################################################
########## Fetching loads connected to a given bus from CIM file ###############
################################################################################
def get_load_info_from_bus(feeder_mrid, bus_name):
    qstr_distributed_load = prefix + """SELECT ?name ?bus ?basev ?p ?q ?conn ?cnt ?pz ?qz ?pi ?qi ?pp ?qp ?pe ?qe ?fdrid (group_concat(distinct ?phs;separator="") as ?phases)
        WHERE
          { 
            ?s r:type c:EnergyConsumer.
            # feeder selection options - if all commented out, query matches all feeders
             VALUES ?fdrid """  +'{"' + str(feeder_mrid) +'"}'+"""
             ?s c:Equipment.EquipmentContainer ?fdr.
             ?fdr c:IdentifiedObject.mRID ?fdrid.
             ?s c:IdentifiedObject.name ?name.
             ?s c:ConductingEquipment.BaseVoltage ?bv.
             ?bv c:BaseVoltage.nominalVoltage ?basev.
             ?s c:EnergyConsumer.customerCount ?cnt.
             ?s c:EnergyConsumer.p ?p.
             ?s c:EnergyConsumer.q ?q.
             ?s c:EnergyConsumer.phaseConnection ?connraw.
               bind(strafter(str(?connraw),"PhaseShuntConnectionKind.") as ?conn)
             ?s c:EnergyConsumer.LoadResponse ?lr.
             ?lr c:LoadResponseCharacteristic.pConstantImpedance ?pz.
             ?lr c:LoadResponseCharacteristic.qConstantImpedance ?qz.
             ?lr c:LoadResponseCharacteristic.pConstantCurrent ?pi.
             ?lr c:LoadResponseCharacteristic.qConstantCurrent ?qi.
             ?lr c:LoadResponseCharacteristic.pConstantPower ?pp.
             ?lr c:LoadResponseCharacteristic.qConstantPower ?qp.
             ?lr c:LoadResponseCharacteristic.pVoltageExponent ?pe.
             ?lr c:LoadResponseCharacteristic.qVoltageExponent ?qe.
             OPTIONAL {?ecp c:EnergyConsumerPhase.EnergyConsumer ?s.
             ?ecp c:EnergyConsumerPhase.phase ?phsraw.
               bind(strafter(str(?phsraw),"SinglePhaseKind.") as ?phs) }
             ?t c:Terminal.ConductingEquipment ?s.
             ?t c:Terminal.ConnectivityNode ?cn. 
             ?cn c:IdentifiedObject.name ?bus
             VALUES ?bus """  +'{"' + str(bus_name) +'"}'+"""
            }
            GROUP BY ?name ?bus ?basev ?p ?q ?cnt ?conn ?pz ?qz ?pi ?qi ?pp ?qp ?pe ?qe ?fdrid
            ORDER by ?name
        """
    sparql.setQuery(qstr_distributed_load)
    ret = sparql.query()
    Property = {}
    for b in ret.bindings:
        for keys in b:
            if 'name' in keys:
                object_name = b[keys].value
                Property[object_name] = {}
            else:
                try:
                    Property[object_name][keys] = float(b[keys].value)
                except:
                    Property[object_name][keys] = (b[keys].value)

    return Property

################################################################################
############ Fetching PVs connected to a given bus from CIM file ###############
################################################################################
def get_PV_info_from_bus(feeder_mrid, bus_name):
    qstr_distributed_PV = prefix + """SELECT ?name ?bus ?ratedS ?ratedU ?ipu ?p ?q ?fdrid (group_concat(distinct ?phs;separator="") as ?phases)
    WHERE {
            ?s r:type c:PhotovoltaicUnit.
             ?s c:IdentifiedObject.name ?name.
             ?pec c:PowerElectronicsConnection.PowerElectronicsUnit ?s.
            # feeder selection options - if all commented out, query matches all feeders
             VALUES ?fdrid """  +'{"' + str(feeder_mrid) +'"}'+"""
             ?pec c:Equipment.EquipmentContainer ?fdr.
             ?fdr c:IdentifiedObject.mRID ?fdrid.
             ?pec c:PowerElectronicsConnection.ratedS ?ratedS.
             ?pec c:PowerElectronicsConnection.ratedU ?ratedU.
             ?pec c:PowerElectronicsConnection.maxIFault ?ipu.
             ?pec c:PowerElectronicsConnection.p ?p.
             ?pec c:PowerElectronicsConnection.q ?q.
             OPTIONAL {?pecp c:PowerElectronicsConnectionPhase.PowerElectronicsConnection ?pec.
             ?pecp c:PowerElectronicsConnectionPhase.phase ?phsraw.
               bind(strafter(str(?phsraw),"SinglePhaseKind.") as ?phs) }
             ?t c:Terminal.ConductingEquipment ?pec.
             ?t c:Terminal.ConnectivityNode ?cn. 
             ?cn c:IdentifiedObject.name ?bus
             VALUES ?bus """  + '{"' + str(bus_name) +'"}'+"""
            }
            GROUP by ?name ?bus ?ratedS ?ratedU ?ipu ?p ?q ?fdrid
            ORDER by ?name
        """
    sparql.setQuery(qstr_distributed_PV)
    ret = sparql.query()
    Property = {}
    for b in ret.bindings:
        for keys in b:
            if 'name' in keys:
                object_name = b[keys].value
                Property[object_name] = {}
            else:
                try:
                    Property[object_name][keys] = float(b[keys].value)
                except:
                    Property[object_name][keys] = (b[keys].value)

    return Property

################################################################################
########### Fetching BESS connected to a given bus from CIM file ###############
################################################################################
def get_BESS_info_from_bus(feeder_mrid, bus_name):
    qstr_distributed_storage = prefix + """SELECT ?name ?bus ?ratedS ?ratedU ?ipu ?ratedE ?storedE ?state ?p ?q ?id ?fdrid (group_concat(distinct ?phs;separator="") as ?phases) 
    WHERE {
            ?s r:type c:BatteryUnit.
             ?s c:IdentifiedObject.name ?name.
             ?pec c:PowerElectronicsConnection.PowerElectronicsUnit ?s.
            # feeder selection options - if all commented out, query matches all feeders
             VALUES ?fdrid """  +'{"' + str(feeder_mrid) +'"}'+"""
             ?pec c:Equipment.EquipmentContainer ?fdr.
             ?fdr c:IdentifiedObject.mRID ?fdrid.
             ?pec c:PowerElectronicsConnection.ratedS ?ratedS.
             ?pec c:PowerElectronicsConnection.ratedU ?ratedU.
             ?pec c:PowerElectronicsConnection.maxIFault ?ipu.
             ?s c:BatteryUnit.ratedE ?ratedE.
             ?s c:BatteryUnit.storedE ?storedE.
             ?s c:BatteryUnit.batteryState ?stateraw.
               bind(strafter(str(?stateraw),"BatteryState.") as ?state)
             ?pec c:PowerElectronicsConnection.p ?p.
             ?pec c:PowerElectronicsConnection.q ?q. 
             OPTIONAL {?pecp c:PowerElectronicsConnectionPhase.PowerElectronicsConnection ?pec.
             ?pecp c:PowerElectronicsConnectionPhase.phase ?phsraw.
               bind(strafter(str(?phsraw),"SinglePhaseKind.") as ?phs) }
             bind(strafter(str(?s),"#_") as ?id).
             ?t c:Terminal.ConductingEquipment ?pec.
             ?t c:Terminal.ConnectivityNode ?cn. 
             ?cn c:IdentifiedObject.name ?bus
             VALUES ?bus """  + '{"' + str(bus_name) +'"}'+"""
            }
            GROUP by ?name ?bus ?ratedS ?ratedU ?ipu ?ratedE ?storedE ?state ?p ?q ?id ?fdrid
            ORDER by ?name
        """
    sparql.setQuery(qstr_distributed_storage)
    ret = sparql.query()
    Property = {}
    for b in ret.bindings:
        for keys in b:
            if 'name' in keys:
                object_name = b[keys].value
                Property[object_name] = {}
            else:
                try:
                    Property[object_name][keys] = float(b[keys].value)
                except:
                    Property[object_name][keys] = (b[keys].value)

    return Property