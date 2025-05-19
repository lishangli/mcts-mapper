import json
class Operation:
    def __init__(self, name, commutative, latency, accumulative, numOperands, numRes, OPC):
        self.name = name
        self.commutative = commutative
        self.latency = latency
        self.acc = accumulative
        self.numOperands = numOperands
        self.numRes = numRes
        self.OPC = OPC

class Operations:
    def __init__(self):
        self.operations = []
        self.name2op = {}
        self.path = ""

    def addOperation(self, operation):
        self.operations.append(operation)
        self.name2op[operation.name] = operation.OPC
        self.name2op[operation.name.lower()] = operation.OPC

    def OpParser(self, jsonFile):
        data = None
        with open(jsonFile) as f:
            data = json.load(f)
        for op in data["Operations"]:
            operation = Operation(op["name"], op["commutative"], op["latency"], op["accumulative"], op["numOperands"], op['numRes'], op["OPC"])
            self.addOperation(operation)

        self.name2op["CONST"]=len(self.name2op)
        self.name2op["const"]=len(self.name2op)
        self.path = jsonFile