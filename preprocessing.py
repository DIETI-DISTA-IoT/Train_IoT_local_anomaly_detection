import random
import torch
import numpy as np

def dict_to_tensor(data_dict):
    # Initialize an empty list to hold numerical values
    values = []
    
    # Iterate through the dictionary
    for key, value in data_dict.items():
        # Check if the value is a number or nan
        if isinstance(value, (int, float)) and not np.isnan(value):
            values.append(value)
        else: 
            # Replace nan with 0 or any other placeholder
            values.append(0.0)  # You can change this to any other placeholder if needed
    
    # Convert the list of values to a PyTorch tensor
    tensor = torch.tensor(values, dtype=torch.float32)
    
    return tensor


class Buffer:
    def __init__(self, size, label=None):
        self.size = size
        self.buffer = []
        self.label = label

    def add(self, item):
        self.buffer.append(item)
        if len(self.buffer) > self.size:
            self.buffer.pop(0)


    def sample(self, n):
        if len(self.buffer) < n:
            record_list = self.buffer
        else:
            record_list = random.sample(self.buffer, n)

        feature_tensors = []
        cluster_labels = []
        class_labels = []

        for record in record_list:
            cluster_label = int(record['cluster'])
            record_copy = record.copy()
            record_copy.pop('cluster', None)
            feature_tensors.append(dict_to_tensor(record_copy))
            cluster_labels.append(cluster_label)
        
        if len(record_list) > 0:
            feature_tensors = torch.stack(feature_tensors)
            cluster_labels = torch.tensor(cluster_labels).unsqueeze(1)
            class_labels = torch.tensor([[self.label]] * len(feature_tensors)).to(torch.float32)

        
        return feature_tensors, class_labels, cluster_labels