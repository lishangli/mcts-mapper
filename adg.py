
from enum import Enum
import numpy as np

class ADGNodeType(Enum):
    GPE = 1 
    IOB = 2

class ADGFeature:
    def __init__(self):
        self.id = None
        self.in_degree = None
        self.type = None # only to support same GPE
        self.out_degree = None
        self.mapped_dfg_id = None

class ADGFeatures:
    def __init__(self, adg):
        self.features = {}
        self.adg = adg
        self.getFeatures()

    def getFeatures(self):
        for id, node in self.adg.getNodes().items():
            if node.getType() == 'GPE':
                feature = ADGFeature()
                feature.id = node.getId()
                feature.in_degree = node.getNumInputs()
                feature.type = 0
                feature.out_degree = node.getNumOutputs()
                feature.mapped_dfg_id = -1
                self.features[feature.id] = feature
            elif node.getType() == 'IOB':
                feature = ADGFeature()
                feature.id = node.getId()
                feature.in_degree = node.getNumInputs()
                feature.type = 1
                feature.out_degree = node.getNumOutputs()
                feature.mapped_dfg_id = -1
                self.features[feature.id] = feature
            elif node.getType() == 'GIB':
                feature = ADGFeature()
                feature.id = node.getId()
                feature.in_degree = node.getNumInputs() # just consider the input port 
                feature.type = 2
                feature.out_degree = node.getNumOutputs() # just consider the output port
                feature.mapped_dfg_id = -1
                self.features[feature.id] = feature

    def getFeaturesVector(self):
        features = np.zeros((self.adg.getMaxNodeId()+1, 5))
        adg_features = self.features
        id = 0
        for idx, feature in adg_features.items():
            features[idx] = [feature.id, feature.type, feature.in_degree, feature.out_degree, feature.mapped_dfg_id]
            id = id + 1
        return features
    
    def __len__(self):
        return len(self.features)
    

"""test code"""
from dfgParser import DFGParser
from adgParser import ADGIR

adgIR = ADGIR("./cgra_adg.json")      
adg = adgIR.getADG()
adgFeatures = ADGFeatures(adg)
adgFeatures.getFeatures()
features = adgFeatures.getFeaturesVector()
print(features)

