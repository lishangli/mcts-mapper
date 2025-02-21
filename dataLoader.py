import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from dfgParser import DFGParser
from dfg import DFGFeatures

from operations import Operation, Operations

class JSONDataset(Dataset):
    def __init__(self, json_dir, ops, transform=None):
        """
        Args:
            json_dir (str): JSON 文件所在目录。
            transform (callable, optional): 用于数据预处理的变换。
        """
        self.json_dir = json_dir
        self.operations = ops
        self.json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        self.transform = transform
        

    def __len__(self):
        """返回数据集大小"""
        return len(self.json_files)

    def __getitem__(self, idx):
        """加载并返回指定索引的数据样本"""
        json_path = os.path.join(self.json_dir, self.json_files[idx])
        parser = DFGParser(json_path)
        dfg = parser.getDFG()
        features = DFGFeatures(dfg).getFeaturesVector(self.operations)
        data = torch.tensor(features, dtype=torch.float32)
        # 加载 JSON 文件
        # with open(json_path, 'r') as f:
        #     data = json.load(f)
        
        # # 假设数据是 {'input': [...], 'label': ...}
        # input_data = torch.tensor(data['input'], dtype=torch.float32)
        # label = torch.tensor(data['label'], dtype=torch.long)
        
        # # 如果有预处理 transform，应用到输入数据
        # if self.transform:
        #     input_data = self.transform(input_data)
        
        return {
            'dfg':dfg,
            'features': data
        }