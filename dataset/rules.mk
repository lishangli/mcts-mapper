## BEGIN INTERFACE ##
BENCHNAME ?= $(shell basename `pwd`)
JSON_DIR = ../json
WAFER_OPT ?= wafer-opt

# 将每个 .dot 文件映射到目标 json 文件路径
# JSON_FILES=$(patsubst ./$(be).dot,$(JSON_DIR)/%.json,$(DOT_FILES))

dfgGen: $(JSON_DIR)/$(BENCHNAME).json

$(JSON_DIR)/$(BENCHNAME).json: func_0.dot
func_0.dot: $(BENCHNAME)-lower.mlir
$(BENCHNAME)-lower.mlir: $(BENCHNAME)-dataflow.mlir
$(BENCHNAME)-dataflow.mlir: $(BENCHNAME).mlir
$(BENCHNAME).mlir: $(BENCHNAME).c

# all: $(BENCHNAME)-lower.mlir

$(BENCHNAME).mlir: $(BENCHNAME).c
	cgeist $< -function=func -S --raise-scf-to-affine > $@

$(BENCHNAME)-dataflow.mlir: $(BENCHNAME).mlir
	@${WAFER_OPT} \
		-symbol-dce -affine-loop-invariant-code-motion\
		-cse -affine-scalrep  --canonicalize\
		--cgra-create-dataflow-from-affine\
		$< -o $@
	echo $@

$(BENCHNAME)-lower.mlir: $(BENCHNAME)-dataflow.mlir
	echo "Generating Dataflow Graph"
	@${WAFER_OPT} $< \
		--cgra-lower-dataflow="cgra-support-ops-path=../operations.json" --canonicalize -o $@

func_0.dot: $(BENCHNAME)-lower.mlir
	@${WAFER_OPT} $< --cgra-dfg-generation -o $@.mlir

$(JSON_DIR)/$(BENCHNAME)-%.json: %.dot
	@mkdir -p $(dir $@)  # 创建目标 json 文件的目录结构
	dot -Tjson $< -o $@  # 执行 dot 命令生成 json 文件
		
clean:
	rm -f *.mlir *.dot