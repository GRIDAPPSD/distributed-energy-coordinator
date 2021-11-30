# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 11:26:10 2021

@author: poud579
"""


"""Module for querying and parsing SPARQL through GridAPPS-D"""
import pandas as pd


class SPARQLManager:
    """Class for querying and parsing SPARQL in GridAPPS-D.
    """

    
    def __init__(self, gapps, feeder_mrid, model_api_topic, simulation_id=None, timeout=30):
        """Connect to the platform.

        :param feeder_mrid: unique identifier for the feeder in
            question. Since PyVVO works on a per feeder basis, this is
            required, and all queries will be executed for the specified
            feeder.
        :param gapps: gridappsd_object
        :param timeout: timeout for querying the blazegraph database.
        """

        # Connect to the platform.
        self.gad = gapps
       
        # Assign feeder mrid.
        self.feeder_mrid = feeder_mrid

        # Timeout for SPARQL queries.
        self.timeout = timeout

        # Powergridmodel API topic
        self.topic = model_api_topic

        # Assign simulation id
        self.simulation_id = simulation_id 

    
    def query_energyconsumer(self):
        """Get information on loads in the feeder."""
        # Perform the query.
        LOAD_QUERY = """
        PREFIX r:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX c:  <http://iec.ch/TC57/CIM100#>
        SELECT ?name ?bus ?basev ?p ?q ?conn ?cnt ?pz ?qz ?pi ?qi ?pp ?qp ?pe ?qe WHERE {
        ?s r:type c:EnergyConsumer.        
        VALUES ?fdrid {"%s"}         
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
        GROUP BY ?name ?bus ?basev ?p ?q ?cnt ?conn ?pz ?qz ?pi ?qi ?pp ?qp ?pe ?qe 
        ORDER by ?name
        """% self.feeder_mrid
        results = self.gad.query_data(LOAD_QUERY)
        bindings = results['data']['results']['bindings']
        list_of_dicts = []
        for obj in bindings:
            list_of_dicts.append({k:v['value'] for (k, v) in obj.items()})
        output = pd.DataFrame(list_of_dicts)
        return output


    def ybus_export(self):
        message = {
        "configurationType": "YBus Export",
        "parameters": {
            "model_id": self.feeder_mrid}
        }

        results = self.gad.get_response("goss.gridappsd.process.request.config", message, timeout=15)
        return results['data']['yParse'],results['data']['nodeList']

    def vnom_export(self):
        message = {
        "configurationType": "Vnom Export",
        "parameters": {
            "model_id": self.feeder_mrid}
        }

        results = self.gad.get_response("goss.gridappsd.process.request.config", message, timeout=15)
        return results['data']['vnom']
