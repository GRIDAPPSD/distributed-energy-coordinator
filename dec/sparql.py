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
    
    def PerLengthPhaseImpedance_line_names(self):
        LINES_QUERY = """
        PREFIX r:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX c:  <http://iec.ch/TC57/CIM100#>
        SELECT ?name ?bus1 ?bus2 ?id (group_concat(distinct ?phs;separator="") as ?phases) WHERE {
        SELECT ?name ?bus1 ?bus2 ?phs ?id WHERE {
        VALUES ?fdrid {"%s"}  # 13 bus
        ?fdr c:IdentifiedObject.mRID ?fdrid.
        ?s r:type c:ACLineSegment.
        ?s c:Equipment.EquipmentContainer ?fdr.
        ?s c:IdentifiedObject.name ?name.
        ?s c:IdentifiedObject.mRID ?id.
        ?t1 c:Terminal.ConductingEquipment ?s.
        ?t1 c:ACDCTerminal.sequenceNumber "1".
        ?t1 c:Terminal.ConnectivityNode ?cn1. 
        ?cn1 c:IdentifiedObject.name ?bus1.
        ?t2 c:Terminal.ConductingEquipment ?s.
        ?t2 c:ACDCTerminal.sequenceNumber "2".
        ?t2 c:Terminal.ConnectivityNode ?cn2. 
        ?cn2 c:IdentifiedObject.name ?bus2
            OPTIONAL {?acp c:ACLineSegmentPhase.ACLineSegment ?s.
            ?acp c:ACLineSegmentPhase.phase ?phsraw.
            bind(strafter(str(?phsraw),"SinglePhaseKind.") as ?phs) }
        } ORDER BY ?name ?phs
        }
        GROUP BY ?name ?bus1 ?bus2 ?id
        ORDER BY ?name
        """% self.feeder_mrid

        results = self.gad.query_data(LINES_QUERY)
        bindings = results['data']['results']['bindings']
        return bindings
    
    def PowerTransformerEnd_xfmr_names(self):
        XFMRS_QUERY = """
        PREFIX r:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX c:  <http://iec.ch/TC57/CIM100#>
        SELECT ?xfmr_name  ?end_number ?bus 
        WHERE {
        VALUES ?fdrid {"%s"}
         ?p c:Equipment.EquipmentContainer ?fdr.
         ?fdr c:IdentifiedObject.mRID ?fdrid.
         ?p r:type c:PowerTransformer.
         ?p c:IdentifiedObject.name ?xfmr_name.
         ?p c:PowerTransformer.vectorGroup ?vector_group.
         ?end c:PowerTransformerEnd.PowerTransformer ?p.
         ?end c:TransformerEnd.endNumber ?end_number.
         ?end c:PowerTransformerEnd.ratedS ?ratedS.
         ?end c:PowerTransformerEnd.ratedU ?ratedU.
         ?end c:PowerTransformerEnd.r ?r_ohm.
         ?end c:PowerTransformerEnd.phaseAngleClock ?angle.
         ?end c:PowerTransformerEnd.connectionKind ?connraw.
          bind(strafter(str(?connraw),"WindingConnection.") as ?connection)
         ?end c:TransformerEnd.grounded ?grounded.
         OPTIONAL {?end c:TransformerEnd.rground ?r_ground.}
         OPTIONAL {?end c:TransformerEnd.xground ?x_ground.}
         ?end c:TransformerEnd.Terminal ?trm.
         ?trm c:Terminal.ConnectivityNode ?cn.
         ?cn c:IdentifiedObject.name ?bus.
         ?end c:TransformerEnd.BaseVoltage ?bv.
         ?bv c:BaseVoltage.nominalVoltage ?base_voltage.
        }
        ORDER BY ?xfmr_name ?end_number
        """% self.feeder_mrid

        results = self.gad.query_data(XFMRS_QUERY)
        bindings = results['data']['results']['bindings']
        return bindings

    def TransformerTank_xfmr_names(self):
        XFMRS_QUERY = """
        PREFIX r:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX c:  <http://iec.ch/TC57/CIM100#>
        SELECT ?xfmr_name ?end_number ?bus ?phase
        WHERE {
        VALUES ?fdrid {"%s"}
         ?p c:Equipment.EquipmentContainer ?fdr.
         ?fdr c:IdentifiedObject.mRID ?fdrid.
         ?p r:type c:PowerTransformer.
         ?p c:IdentifiedObject.name ?pname.
         ?p c:PowerTransformer.vectorGroup ?vector_group.
         ?t c:TransformerTank.PowerTransformer ?p.
         ?t c:IdentifiedObject.name ?xfmr_name.
         ?asset c:Asset.PowerSystemResources ?t.
         ?asset c:Asset.AssetInfo ?inf.
         ?inf c:IdentifiedObject.name ?xfmr_code.
         ?end c:TransformerTankEnd.TransformerTank ?t.
         ?end c:TransformerTankEnd.phases ?phsraw.
          bind(strafter(str(?phsraw),"PhaseCode.") as ?phase)
         ?end c:TransformerEnd.endNumber ?end_number.
         ?end c:TransformerEnd.grounded ?grounded.
         OPTIONAL {?end c:TransformerEnd.rground ?rground.}
         OPTIONAL {?end c:TransformerEnd.xground ?xground.}
         ?end c:TransformerEnd.Terminal ?trm.
         ?trm c:Terminal.ConnectivityNode ?cn.
         ?cn c:IdentifiedObject.name ?bus.
         ?end c:TransformerEnd.BaseVoltage ?bv.
         ?bv c:BaseVoltage.nominalVoltage ?baseV.
        }
        ORDER BY ?xfmr_name ?end_number
        """% self.feeder_mrid

        results = self.gad.query_data(XFMRS_QUERY)
        bindings = results['data']['results']['bindings']
        return bindings

    def TransformerTank_xfmr_r(self):
        XMFRS_QUERY = """
        PREFIX r:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX c:  <http://iec.ch/TC57/CIM100#>
        SELECT ?xfmr_name ?xfmr_code ?enum ?ratedS ?ratedU ?connection ?angle ?r_ohm
        WHERE {
        VALUES ?fdrid {"%s"}
         ?eq c:Equipment.EquipmentContainer ?fdr.
         ?fdr c:IdentifiedObject.mRID ?fdrid.
         ?xft c:TransformerTank.PowerTransformer ?eq.
         ?xft c:IdentifiedObject.name ?xfmr_name.
         ?asset c:Asset.PowerSystemResources ?xft.
         ?asset c:Asset.AssetInfo ?t.
         ?p r:type c:PowerTransformerInfo.
         ?t c:TransformerTankInfo.PowerTransformerInfo ?p.
         ?t c:IdentifiedObject.name ?tname.
         ?t c:IdentifiedObject.mRID ?id.
         ?e c:TransformerEndInfo.TransformerTankInfo ?t.
         ?e c:IdentifiedObject.mRID ?eid.
         ?e c:IdentifiedObject.name ?xfmr_code.
         ?e c:TransformerEndInfo.endNumber ?enum.
         ?e c:TransformerEndInfo.ratedS ?ratedS.
         ?e c:TransformerEndInfo.ratedU ?ratedU.
         ?e c:TransformerEndInfo.r ?r_ohm.
         ?e c:TransformerEndInfo.phaseAngleClock ?angle.
         ?e c:TransformerEndInfo.connectionKind ?connraw.
          bind(strafter(str(?connraw),"WindingConnection.") as ?connection)
        }
        ORDER BY ?xfmr_name ?xfmr_code ?enum
        """% self.feeder_mrid

        results = self.gad.query_data(XMFRS_QUERY)
        bindings = results['data']['results']['bindings']
        return bindings
    
    def TransformerTank_xfmr_z(self):
        XMFRS_QUERY = """
        PREFIX r:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX c:  <http://iec.ch/TC57/CIM100#>
        SELECT ?xfmr_name ?enum ?gnum ?leakage_z ?loadloss
        WHERE {
        VALUES ?fdrid {"%s"}
         ?eq c:Equipment.EquipmentContainer ?fdr.
         ?fdr c:IdentifiedObject.mRID ?fdrid.
         ?xft c:TransformerTank.PowerTransformer ?eq.
         ?xft c:IdentifiedObject.name ?xfmr_name.
         ?asset c:Asset.PowerSystemResources ?xft.
         ?asset c:Asset.AssetInfo ?t.
         ?p r:type c:PowerTransformerInfo.
         ?t c:TransformerTankInfo.PowerTransformerInfo ?p.
         ?e c:TransformerEndInfo.TransformerTankInfo ?t.
         ?e c:TransformerEndInfo.endNumber ?enum.
         ?sct c:ShortCircuitTest.EnergisedEnd ?e.
         ?sct c:ShortCircuitTest.leakageImpedance ?leakage_z.
         ?sct c:ShortCircuitTest.loss ?loadloss.
         ?sct c:ShortCircuitTest.GroundedEnds ?grnd.
         ?grnd c:TransformerEndInfo.endNumber ?gnum.
        }
        ORDER BY ?xfmr_name ?enum
        """% self.feeder_mrid

        results = self.gad.query_data(XMFRS_QUERY)
        bindings = results['data']['results']['bindings']
        return bindings      

    def SwitchingEquipment_switch_names(self):
        SWITCHES_QUERY = """
        PREFIX r:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX c:  <http://iec.ch/TC57/CIM100#>
        SELECT ?cimtype ?name ?isopen ?bus1 ?bus2 ?id (group_concat(distinct ?phs;separator="") as ?phases) WHERE {
        SELECT ?cimtype ?name ?isopen ?bus1 ?bus2 ?phs ?id WHERE {
        VALUES ?fdrid {"%s"}  # 13 bus
        VALUES ?cimraw {c:LoadBreakSwitch c:Recloser c:Breaker}
        ?fdr c:IdentifiedObject.mRID ?fdrid.
        ?s r:type ?cimraw.
        bind(strafter(str(?cimraw),"#") as ?cimtype)
        ?s c:Equipment.EquipmentContainer ?fdr.
        ?s c:IdentifiedObject.name ?name.
        ?s c:IdentifiedObject.mRID ?id.
        ?s c:Switch.normalOpen ?isopen.
        ?t1 c:Terminal.ConductingEquipment ?s.
        ?t1 c:ACDCTerminal.sequenceNumber "1".
        ?t1 c:Terminal.ConnectivityNode ?cn1. 
        ?cn1 c:IdentifiedObject.name ?bus1.
        ?t2 c:Terminal.ConductingEquipment ?s.
        ?t2 c:ACDCTerminal.sequenceNumber "2".
        ?t2 c:Terminal.ConnectivityNode ?cn2. 
        ?cn2 c:IdentifiedObject.name ?bus2
            OPTIONAL {?swp c:SwitchPhase.Switch ?s.
            ?swp c:SwitchPhase.phaseSide1 ?phsraw.
            bind(strafter(str(?phsraw),"SinglePhaseKind.") as ?phs) }
        } ORDER BY ?name ?phs
        }
        GROUP BY ?cimtype ?name ?isopen ?bus1 ?bus2 ?id
        ORDER BY ?cimtype ?name
        """% self.feeder_mrid

        results = self.gad.query_data(SWITCHES_QUERY)
        bindings = results['data']['results']['bindings']
        return bindings

    def query_energyconsumer_lf(self):
        """Get information on loads in the feeder."""
        # Perform the query.
        LOAD_QUERY = """
        PREFIX r:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX c:  <http://iec.ch/TC57/CIM100#>
        SELECT ?name ?bus ?p ?q (group_concat(distinct ?phs;separator="\\n") as ?phases) WHERE {
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
        GROUP BY ?name ?bus ?p ?q 
        ORDER by ?name
        """% self.feeder_mrid
        results = self.gad.query_data(LOAD_QUERY)
        bindings = results['data']['results']['bindings']
        return bindings
    
    def query_photovoltaic(self):
        """Get information on loads in the feeder."""
        # Perform the query.
        PV_QUERY = """
        PREFIX r:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX c:  <http://iec.ch/TC57/CIM100#>
        SELECT ?name ?bus ?ratedS ?ratedU ?ipu ?p ?q ?fdrid (group_concat(distinct ?phs;separator="\\n") as ?phases) WHERE {
        ?s r:type c:PhotovoltaicUnit.
        ?s c:IdentifiedObject.name ?name.
        ?pec c:PowerElectronicsConnection.PowerElectronicsUnit ?s.
        VALUES ?fdrid {"%s"}  # 123 bus
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
        }
        GROUP by ?name ?bus ?ratedS ?ratedU ?ipu ?p ?q ?fdrid
        ORDER by ?name
        """% self.feeder_mrid
        results = self.gad.query_data(PV_QUERY)
        bindings = results['data']['results']['bindings']
        return bindings


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


