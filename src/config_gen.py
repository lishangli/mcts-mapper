from cgra_compiler_python import *

ops = Operations.Instance("operations.json")
air = ADGIR("cgra_adg.json")
dir = DFGIR("dfg.json")

adg = air.getADG()
dfg = dir.getDFG()

map = MapperSA(adg, dfg, 0, 0, True)

map.executeBind(True, False, ".", "cgra_execute", "config.json")