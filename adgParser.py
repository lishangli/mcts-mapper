import json
import copy

### GRAPH DEFINITION
class Graph:
    def __init__(self):
        self._id = None
        self._bitWidth = None
        self._inputNames = {}
        self._outputNames = {}
        self._inputs = {}
        self._outputs = {}
        self._inputEdges = {}
        self._outputEdges = {}

    def setId(self, id):
        self._id = id
    
    def getId(self):
        return self._id
    
    def setBitWidth(self, bitWidth):
        self._bitWidth = bitWidth

    def getBitWidth(self):
        return self._bitWidth
    
    def setInputNames(self, inputNames):
        self._inputNames = inputNames

    def getInputNames(self):
        return self._inputNames
    
    def setOutputNames(self, outputNames):
        self._outputNames = outputNames

    def getOutputNames(self):
        return self._outputNames

    def getInputs(self):
        return self._inputs
    
    def getNumInputs(self):
        return len(self._inputs)
    
    def getInput(self, index):
        return self._inputs[index]
    
    def getOutput(self,index):
        return self._outputs[index]

    def getNumOutputs(self):
        return len(self._outputs)

    def getOutputs(self):
        return self._outputs
    
    def addInput(self, index, input):
        self._inputs[index] = input

    def addOutput(self,index, output):
        self._outputs[index] = output

    def printGraph(self):
        print("===============================================================")
        print("id: {}\n".format(self._id))
        print("bitWidth: {}\n".format(self._bitWidth))
        print("inputNames: {}\n".format(self._inputNames))
        print("outputNames: {}\n".format(self._outputNames))
        print("numInputs: {}\n".format(len(self._inputs)))
        print("numOutputs: {}\n".format(len(self._outputs)))
        print("inputs: {}\n".format(self._inputs))
        print("outputs: {}\n".format(self._outputs))


class GraphNode:
    def __init__(self, id):
        self._id = id
        self._name = None
        self._type = None
        self._bitWidth = None
        self._inputs = {}
        self._outputs = {}
        self._inputEdges = {}
        self._outputEdges = {}

    def setId(self, id):
        self._id = id

    def getId(self):
        return self._id
    
    def setName(self, name):
        self._name = name

    def getName(self):
        return self._name
    
    def setType(self, type):
        self._type = type

    def getType(self):
        return self._type

    def setBitWidth(self, bitWidth):
        self._bitWidth = bitWidth

    def getBitWidth(self):
        return self._bitWidth
    
    def setInputs(self, inputs):
        self._inputs = inputs

    def getInputs(self):
        return self._inputs
    
    def setOutputs(self, outputs):
        self._outputs = outputs

    def getOutputs(self):
        return self._outputs
    
    def getInput(self, index):
        return self._inputs[index]
    
    def getOutput(self, index):
        return self._outputs[index]
    
    def getNumInputs(self):
        return len(self._inputs)
    
    def getNumOutputs(self):
        return len(self._outputs)
    
    def addInput(self, index, input):
        self._inputs[index] = input

    def addOutput(self, index, output):
        self._outputs[index] = output

    def addInputEdge(self, index, edge):
        self._inputEdges[index] = edge

    def addOutputEdge(self, index, edge):
        self._outputEdges[index] = edge

    def delInputEdge(self, index):
        if index in self._inputEdges:
            del self._inputEdges[index]
    
    def delOutputEdge(self, index):
        if index in self._outputEdges:
            del self._outputEdges[index]
    
    def printGraphNode(self):
        print("id: {}\n".format(self._id))
        print("name: {}\n".format(self._name))
        print("type: {}\n".format(self._type))
        print("bitWidth: {}\n".format(self._bitWidth))
        print("inputs: {}\n".format(self._inputs))
        print("outputs: {}\n".format(self._outputs))
    
class GrpahEdge:
    def __init__(self, srcId, dstId):
        self._id = None
        self._srcPortIdx = None 
        self._dstPortIdx = None
        if srcId is not None and dstId is not None:
            self._srcId = srcId
            self._dstId = dstId
        elif srcId is not None and dstId is None:
            self._id = srcId

    def setId(self, id):
        self._id = id

    def getId(self):
        return self._id
    
    def setSrcPortIdx(self, srcPortIdx):
        self._srcPortIdx = srcPortIdx

    def getSrcPortIdx(self):
        return self._srcPortIdx
    
    def setDstPortIdx(self, dstPortIdx):
        self._dstPortIdx = dstPortIdx
    
    def getDstPortIdx(self):
        return self._dstPortIdx
    
    def setSrcId(self, srcId):
        self._srcId = srcId

    def getSrcId(self):
        return self._srcId
    
    def setDstId(self, dstId):
        self._dstId = dstId

    def getDstId(self):
        return self._dstId
    
    def setEdge(self, srcId, dstId):
        self._srcId = srcId
        self._dstId = dstId

    def setEdge4(self, srcId, srcPortIdx, dstId, dstPortIdx):
        self._srcId = srcId
        self._srcPortIdx = srcPortIdx
        self._dstId = dstId
        self._dstPortIdx = dstPortIdx

### Config DATA DEFINITION
class CfgDataLoc:
    def __init__(self):
        low = None 
        high = None

### Arch Description Graph DEFINITION    
class ADGNode(GraphNode):
    def __init__(self, id):
        super().__init__(id)
        self._cfgBlkIdx =None
        self._x = None
        self._y = None
        self._subADG = None 
        self._cfgBits = {}
        self._configInfo = {}
        self.useInPort = set()
        self.useOutPort = set()

    def getCfgBlkIdx(self):
        return self._cfgBlkIdx
    
    def setCfgBlkIdx(self,cfgBlkIdx):
        self._cfgBlkIdx = cfgBlkIdx

    def getX(self):
        return self._x
    
    def setX(self,x):
        self._x = x

    def getY(self):
        return self._y
    
    def setY(self,y):
        self._y = y

    def getSubADG(self):
        return self._subADG
    
    def setSubADG(self,subADG):
        self._subADG = subADG

    def getConfigInfo(self, *args, **kwargs):
        if (len(args) == 0):
            return self._configInfo
        else:
            return self._configInfo[args[0]]
    
    def addConfigInfo(self,id, subModuleCfg):
        self._configInfo[id] = subModuleCfg

    def printADGNode(self):
        super().printGraphNode()
        print ("cfgBlkIdx: {}\n".format(self._cfgBlkIdx))


class FUNode(ADGNode):
    def __init__(self, id):
        super().__init__(id)
        self._numOperands = None
        self._maxDelay = None
        self._operandInputs = {}
        self._numRfReg = None
        self._operations = set()
        self.cfgIdMap = {}

    def getNumOperands(self):
        return self._numOperands
    
    def setNumOperands(self,numOperands):
        self._numOperands = numOperands

    def getMaxDelay(self):
        return self._maxDelay
    
    def setMaxDelay(self,maxDelay):
        self._maxDelay = maxDelay

    def getOperandInputs(self, opeIdx):
        return self._operandInputs[opeIdx]
    
    def addOperandInputs(self, opeIdx, operandIdx):
        if (isinstance(operandIdx, int)):
            if (opeIdx not in self._operandInputs):
                self._operandInputs[opeIdx] = operandIdx
        elif (isinstance(operandIdx, list)): 
                self._operandInputs[opeIdx] = operandIdx
        else: 
            raise ValueError("operandIdx should be either int or list")
        
    def delOperandInputs(self, opeIdx, operandIdx):
        assert len(self._operandInputs) > opeIdx 
        if (self._operandInputs[opeIdx] != None):
            self._operandInputs[opeIdx].remove(operandIdx)

    def addOperation(self, operation):
        self._operations.add(operation)

    def delOperation(self, operation):
        self._operations.discard(operation)

    def getOperations(self):
        self._operations

    def OpCapable(self, operation):
        return operation in self._operations
    
    def getOperandIdx(self, inputIdx):
        return self._operandInputs.index(inputIdx)
    
    def printFU(self):
        super().printADGNode()
        print ("numOperands: {}\n".format(self._numOperands))
        print ("maxDelay: {}\n".format(self._maxDelay))
        print ("operandInputs: {}\n".format(self._operandInputs))
        print ("cfgIdMap: {}\n".format(self.cfgIdMap))

    

class GPENode(FUNode):
    def __init__(self, id):
        super().__init__(id)
        self._numRfReg = None

    def getNumRfReg(self):
        return self._numRfReg
    
    def setNumRfReg(self,numRfReg):
        self._numRfReg = numRfReg

    def printGPENode(self):
        super().printFU()
        print("numRfReg: {}\n".format(self._numRfReg))

class IOBNode(FUNode):
    def __init__(self, id):
        super().__init__(id)
        self._index = None
        self._mode = None 

    def getIndex(self):
        return self._index

    def setIndex(self,index):
        self._index = index

    def printIOBNode(self):
        super().printFU()
        print("index: {}\n".format(self._index))
        print("mode: {}\n".format(self._mode))


class GIBNode(FUNode):
    def __init__(self, id):
        super().__init__(id)
        self._outReged = {}
        self._out2ins = {}
        self._in2outs = {}
        self._trackReged = None

    def getTrackReged(self):
        return self._trackReged
    
    def setTrackReged(self,trackReged):
        self._trackReged = trackReged

    def getOutReged(self, idx):
        return self._outReged[idx]
    
    def setOutReged(self,idx, outReged):
        self._outReged[idx] = outReged

    def getOut2Ins(self, outPort):
        if outPort not in self._out2ins:
            return {}
        return self._out2ins[outPort]
    
    def getIn2Outs(self, inPort):
        if inPort not in self._in2outs:
            return {}
        return self._in2outs[inPort]
    
    def addOut2Ins(self, outPort, inPort):
        if self._inputs[inPort] != None and self._outputs[outPort] != None:
            self._out2ins[outPort] = inPort

    def addIn2Outs(self, inPort, outPort):
        if self._inputs[inPort] != None and self._outputs[outPort] != None:
            self._in2outs[inPort] = outPort

    def printGIBNode(self):
        super().printADGNode()
        print("trackReged: {}\n".format(self._trackReged))
        print("outReged: {}\n".format(self._outReged))
        print("out2ins: {}\n".format(self._out2ins))
        print("in2outs: {}\n".format(self._in2outs))


class ADG(Graph):
    def __init__(self):
        super().__init__()
        self._numGpeNodes = None
        self._numIoNodes = None
        self._cfgDataWidth = None
        self._cfgAddrWidth = None
        self._cfgBlkOffset = None
        self._cfgSpadDataWidth = None
        self._cfgSpadSize = None
        self._iobAgNestLevels = None 
        self._iobSpadBankSize = None

        self._iobToSpadBanks = {}
        self._cfgBits = {}
        self._nodes = {}
        self._edges = {}

    def getNumGpeNodes(self):
        return self._numGpeNodes

    def setNumGpeNodes(self,numGpeNodes):
        self._numGpeNodes = numGpeNodes

    def getNumIoNodes(self):
        return self._numIoNodes

    def setNumIoNodes(self,numIoNodes):
        self._numIoNodes = numIoNodes

    def getCfgDataWidth(self):
        return self._cfgDataWidth

    def setCfgDataWidth(self,cfgDataWidth):
        self._cfgDataWidth = cfgDataWidth

    def getCfgAddrWidth(self):
        return self._cfgAddrWidth
    
    def setCfgAddrWidth(self,cfgAddrWidth):
        self._cfgAddrWidth = cfgAddrWidth

    def getCfgBlkOffset(self):
        return self._cfgBlkOffset
    
    def setCfgBlkOffset(self,cfgBlkOffset):
        self._cfgBlkOffset = cfgBlkOffset

    def getCfgSpadDataWidth(self):
        return self._cfgSpadDataWidth   
    
    def setCfgSpadDataWidth(self,cfgSpadDataWidth):
        self._cfgSpadDataWidth = cfgSpadDataWidth

    def setCfgSpadSize(self,cfgSpadSize):
        self._cfgSpadSize = cfgSpadSize

    def getCfgSpadSize(self):
        return self._cfgSpadSize

    def iobAgNestLevels(self):
        return self._iobAgNestLevels
    
    def setIobAgNestLevels(self,iobAgNestLevels):
        self._iobAgNestLevels = iobAgNestLevels

    def getIobSpadBankSize(self):
        return self._iobSpadBankSize
    
    def setIobSpadBankSize(self,iobSpadBankSize):
        self._iobSpadBankSize = iobSpadBankSize

    def getIobToSpadBanks(self, *args, **kwargs):
        if (len(args) == 0):
            return self._iobToSpadBanks
        else:
            return self._iobToSpadBanks[args[0]]
        
    def setIobToSpadBanks(self,iobIdx, spadIdx):
        self._iobToSpadBanks[iobIdx] = spadIdx

    def getNodes(self):
        return self._nodes
    
    def getEdges(self):
        return self._edges
    
    def getNode(self, id):
        return self._nodes[id]
    
    def getEdge(self):
        return self._edges[id]
    
    def addNode(self, id, node):
        self._nodes[id] = node

    def addEdge(self, edge):
        id = edge.getId()
        self._edges[id] = edge
        srcId = edge.getSrcId()
        dstId = edge.getDstId()
        srcPort = edge.getSrcPortIdx()
        dstPort = edge.getDstPortIdx()
        if srcId == self._id :
            super().addInput(srcPort, (dstId, dstPort))
        else:
            src = self.getNode(srcId)
            if src:
                src.addOutput(srcPort, (dstId, dstPort))

        if dstId == self._id:
            super().addOutput(dstPort, (srcId, srcPort))
        else:
            dst = self.getNode(dstId)
            if dst:
                dst.addInput(dstPort, (srcId, srcPort))

    def getNode(self, id):
        if id not in self._nodes:
            return None
        return self._nodes[id]
    
    def getMaxNodeId(self):
        maxId = 0
        for id in self._nodes:
            maxId = max(maxId, id)
        return maxId
    
    def getEdge(self, id):
        if id not in self._edges:
            raise ValueError("Edge {} not found".format(id))
        return self._edges[id]

    def delNode(self, id):
        del self._nodes[id]

    def delEdge(self, id):
        del self._edges[id]

    def printADG(self):
        print("numGpeNodes: {}\n".format(self._numGpeNodes))
        print("numIoNodes: {}\n".format(self._numIoNodes))
        print("cfgDataWidth: {}\n".format(self._cfgDataWidth))
        print("cfgAddrWidth: {}\n".format(self._cfgAddrWidth))
        print("cfgBlkOffset: {}\n".format(self._cfgBlkOffset))
        print("cfgSpadDataWidth: {}\n".format(self._cfgSpadDataWidth))
        print("cfgSpadSize: {}\n".format(self._cfgSpadSize))
        print("iobAgNestLevels: {}\n".format(self._iobAgNestLevels))
        print("iobSpadBankSize: {}\n".format(self._iobSpadBankSize))
        print("iobToSpadBanks: {}\n".format(self._iobToSpadBanks))
        # for node in self._nodes.values():
        #     if isinstance(node, GPENode):
        #         node.printGPENode()
        #     elif isinstance(node, IOBNode):
        #         node.printIOBNode()
        #     elif isinstance(node, GIBNode):
        #         node.printGIBNode()
        #     else:
        #         raise ValueError("Node type not found")
        # print("nodes: {}\n".format(self._nodes))
        # print("edges: {}\n".format(self._edges))

### Arch Description Graph Definition IR
class ADGIR :
    def __init__(self, jsonFile) -> None:
        data = None 
        self._iobModeNames = {}
        with open(jsonFile) as f:
            data = json.load(f)
            self._adg = self.parseADG(data)

    def getADG(self):
        return self._adg
    
    def parseADG(self, data):

        adg = ADG()
        adg.setBitWidth(data["data_width"])

        if "cfg_spad_data_width" in data:
            adg.setCfgSpadDataWidth(data["cfg_spad_data_width"])

        if "cfg_data_width" in data:
            adg.setCfgDataWidth(data["cfg_data_width"])
            adg.setCfgAddrWidth(data["cfg_addr_width"])
            adg.setCfgBlkOffset(data["cfg_blk_offset"])
        
        if "iob_ag_nest_levels" in data:
            adg.setIobAgNestLevels(data["iob_ag_nest_levels"])

        if "iob_mode_names" in data:
            for mode, name in data["iob_mode_names"].items():
                self._iobModeNames[int(mode)] = name

        if "iob_spad_bank_size" in data:
            adg.setIobSpadBankSize(data["iob_spad_bank_size"])
        
        if "iob_to_spad_banks" in data:
            for iobIdx, spadIdx in data["iob_to_spad_banks"].items():
                adg.setIobToSpadBanks(iobIdx, spadIdx)

        if "cfg_spad_size" in data:
            adg.setCfgSpadSize(data["cfg_spad_size"])

        modules = {}
        for nodeJson in data["sub_modules"] :
            node = self.parseADGNode(nodeJson)
            modules[node.getId()] = (node, False)

        for nodeJson in data["instances"]:
            node = self.parseADGModuleNode(nodeJson, modules)
            nodeId = nodeJson["id"]
            if node :
                adg.addNode(nodeId, node)
            else :
                adg.setId(nodeId)
        
        self.parseADGEdges(adg, data["connections"])
        self.postProcessADG(adg)

        #self._adg = adg
        return adg
    
    def parseADGNode(self, nodeJson):
        type = nodeJson["type"]
        nodeId = nodeJson["id"]

        adg_node = ADGNode(nodeId)

        if type == "GPE" or type == "GIB" or type == "IOB" :
            attrs = nodeJson["attributes"]
            if type == "GPE" or type == "IOB":
                fu_node = None
                if type == "GPE":
                    node = GPENode(nodeId)
                    for op in attrs["operations"]:
                        node.addOperation(op)
                    
                    if "affine_ctrl_reg_cfg_id" in attrs: 
                        iocCfgId = attrs["affine_ctrl_reg_cfg_id"]
                        node.cfgIdMap["InitVal"] = iocCfgId["InitVal"]
                        node.cfgIdMap["Cycles"] = iocCfgId["Cycles"]
                        node.cfgIdMap["WI"] = iocCfgId["WI"]
                        node.cfgIdMap["Latency"] = iocCfgId["Latency"]
                        node.cfgIdMap["Repeats"] = iocCfgId["Repeats"]
                        node.cfgIdMap["SkipFirst"] = iocCfgId["SkipFirst"]
                    
                    fu_node = node
                
                else :
                    node = IOBNode(nodeId)
                    iocCfgId = attrs["io_controller_cfg_id"]
                    node.cfgIdMap["BaseAddr"] = iocCfgId["BaseAddr"]
                    node.cfgIdMap["II"] = iocCfgId["II"]
                    node.cfgIdMap["Latency"] = iocCfgId["Latency"]
                    node.cfgIdMap["IsStore"] = iocCfgId["IsStore"]
                    if "UseAddr" in iocCfgId:
                        node.cfgIdMap["UseAddr"] = iocCfgId["UseAddr"]

                    agNestLevels = attrs["ag_nest_levels"]
                    for i in range(agNestLevels):
                        stringName = "Stride" + str(i)
                        node.cfgIdMap[stringName] = iocCfgId[stringName]
                        cycleName = "Cycles" + str(i)
                        node.cfgIdMap[cycleName] = iocCfgId[cycleName]
                    iobMode = attrs["iob_mode"]
                    modeName = self._iobModeNames[iobMode]
                    if modeName == "FIFO_MODE":
                        node.addOperation("INPUT")
                        node.addOperation("OUTPUT")
                    else :## modeName == "SRAM_MODE"
                        node.addOperation("LOAD")
                        node.addOperation("STORE")
                        node.addOperation("INPUT")
                        node.addOperation("OUTPUT") 

                    fu_node = node

                if "max_delay" in attrs:
                    fu_node.setMaxDelay(attrs["max_delay"])
                else :
                    fu_node.setMaxDelay(0)

                fu_node.setNumOperands(attrs["num_operands"])
                adg_node = fu_node

            elif type == "GIB":
                node = GIBNode(nodeId)
                adg_node = node

            adg_node.setCfgBlkIdx(attrs["cfg_blk_index"])
            subADG = self.parseADG(attrs)
            adg_node.setSubADG(subADG)

            if "configuration" in attrs:
                for subModuleId, info in attrs["configuration"].items():
                    cfg = CfgDataLoc()
                    cfg.low = info[1]
                    cfg.high = info[2]
                    adg_node.addConfigInfo(subModuleId, cfg)
        else:
            adg_node = ADGNode(nodeId)

        adg_node.setType(type)
        return adg_node

    def parseADGModuleNode(self, nodeJson, modules):
        type = nodeJson["type"]
        if type == "This":
            return None
        
        nodeId = nodeJson["id"]
        moduleId = nodeJson["module_id"]

        adg_node = None 
        module = (modules[moduleId])[0]
        renewNode = (modules[moduleId])[1]
        if type == "GPE" or type == "GIB" or type == "IOB":
            if renewNode:
                if type == "GPE":
                    node = GPENode(nodeId)
                    node = copy.copy(module)
                    adg_node = node
                elif type == "IOB":
                    node = IOBNode(nodeId)
                    node = copy.copy(module)
                    adg_node = node
                else:
                    node = GIBNode(nodeId)
                    node = copy.copy(module)
                    adg_node = node
                subADG = ADG()
                subADG = copy.copy(module.getSubADG())
                adg_node.setSubADG(subADG)

            else:
                adg_node = module
                (modules[moduleId]) = (adg_node, True)
            if type == "GPE":
                adg_node.setMaxDelay(nodeJson["max_delay"])
            elif type == "IOB":
                adg_node.setIndex(nodeJson["iob_index"])
                adg_node.setMaxDelay(nodeJson["max_delay"])
            else:
                adg_node.setTrackReged(nodeJson["track_reged"])

            adg_node.setCfgBlkIdx(nodeJson["cfg_blk_index"])
            adg_node.setX(nodeJson["x"])
            adg_node.setY(nodeJson["y"])
        else:
            if renewNode:
                node = ADGNode(nodeId)
                node = copy.copy(module)
                adg_node = node
            else:
                adg_node = module
                (modules[moduleId]) = (adg_node,True)
        adg_node.setId(nodeId)
        adg_node.setName(type + str(nodeId))

        return adg_node
    
    def parseADGEdges(self, adg, nodeJson):
        for key, value in nodeJson.items():
            edgeId = key 
            edge = value 
            srcId = edge[0]
            srcPort = edge[2]
            dstId = edge[3]
            dstPort = edge[5]
            adg_edge = GrpahEdge(srcId, dstId)
            adg_edge.setSrcPortIdx(srcPort)
            adg_edge.setDstPortIdx(dstPort)
            adg_edge.setId(edgeId)
            adg_edge.setSrcId(srcId)
            adg_edge.setDstId(dstId)
            adg.addEdge(adg_edge)
            

    def analyzeIntraConnect(self, node):
        subAdg = node.getSubADG()
        if type(node) == GPENode:
            for nodeId, input in subAdg.getInputs().items():
                subNode = subAdg.getNode(input[0])
                opeIdx = input[1]

                while subNode.getType() != "ALU":
                    if len(subNode.getOutputs()) == 1:
                        opeIdx = 0
                    out = subNode.getOutput(opeIdx)
                    subNode = subAdg.getNode(out[0])
                    opeIdx = out[1]
                node.addOperandInputs(opeIdx, nodeId)
        elif type(node) == IOBNode:
            for nodeId, input in subAdg.getInputs().items():
                subNode = subAdg.getNode(input[0])
                opeIdx = input[1]

                while subNode.getType() != "IOController":
                    if len(subNode.getOutputs()) == 1:
                        opeIdx = 0
                    out = subNode.getOutput(opeIdx)
                    subNode = subAdg.getNode(out[0])
                    opeIdx = out[1]
                node.addOperandInputs(opeIdx, nodeId)
        elif type(node) == GIBNode:
            for inPort, subNode in subAdg.getInputs().items():
                outPort = None 
                if subNode[0] == subAdg.getId():
                    outPort = subNode[1]
                else:
                    subAdgNode = subAdg.getNode(subNode[0])
                    outPort = subAdgNode.getOutput(0)[1]

                node.addIn2Outs(inPort, outPort)
                node.addOut2Ins(outPort, inPort)

    def analyzeOutReg(self, adg, node):
        for id, output in node.getOutputs().items():
            nodeId = (output[0])
            if adg.getNode(nodeId).getType() == "GIB" and node.getTrackReged() != None:
                node.setOutReged(id, True)
            else:
                node.setOutReged(id, False)


    def postProcessADG(self,adg):
        numGpeNodes = 0
        numIobNodes = 0
        for id, node in adg.getNodes().items():
            if node.getType() == "GPE":
                numGpeNodes += 1
                self.analyzeIntraConnect(node)
            elif node.getType() == "IOB":
                numIobNodes += 1
                self.analyzeIntraConnect(node)
            elif node.getType() == "GIB":
                self.analyzeIntraConnect(node)
                self.analyzeOutReg(adg, node)
        adg.setNumGpeNodes(numGpeNodes)
        adg.setNumIoNodes(numIobNodes)



# ## Test
# adgIR = ADGIR("./cgra_adg.json")                   

# adgIR.getADG().printADG()       

