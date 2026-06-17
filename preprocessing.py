import random
import torch
torch.backends.mkldnn.enabled = False
torch.backends.nnpack.enabled = False
import numpy as np
from threading import Lock


def dict_to_tensor(data_dict):
    
    uncampled_values = [ (value  if isinstance(value, (int, float)) and not np.isnan(value) else 0.0) for value in data_dict.values() ]

    tensor = torch.tensor(uncampled_values, dtype=torch.float32) 

    ## ALTERNATIVE 1: convert to numpy clamping the inf values:
    # uncampled_values = np.array(uncampled_values, dtype=np.float32)
    # clampled_values = np.clip(uncampled_values, a_min=-1e6, a_max=1e6)
    # tensor = torch.tensor(clampled_values, dtype=torch.float32)

    ## ALTERNATIVE 2: convert to list clamping the inf values:
    # clampled_values = [ max(min(value, 30000), -4000) for value in uncampled_values ]
    # tensor = torch.tensor(clampled_values, dtype=torch.float32)

    ## ALTERNATIVE 3: convert to list clamping the inf values:
    # tensor = torch.tensor(uncampled_values, dtype=torch.float32)
    # tensor = torch.nan_to_num(tensor, posinf=1e6, neginf=-1e6)

    return tensor


class Buffer:
    def __init__(self, size):
        self.size = size
        self.feats = []
        self.main_labels = []
        self.lock = Lock()


    def add(self, feat_tensor, main_label_tensor):
            
            with self.lock:
                self.feats.append(feat_tensor)
                self.main_labels.append(main_label_tensor)
                if len(self.feats) > self.size:
                    self.feats.pop(0)
                    self.main_labels.pop(0)


    def format(self, item):

        main_label = torch.tensor(item['event_type'], dtype=torch.long)
        del item['event_type']
        feat_tensor = dict_to_tensor(item)
        return feat_tensor,  main_label

    

    def sample(self, n):
        
        feats = []
        main_labels = []

        with self.lock:
            if len(self.feats) < n:
                feats = self.feats
                main_labels = self.main_labels
            else:
                feats = random.sample(self.feats, n)
                main_labels = random.sample(self.main_labels, n)

        if len(feats) > 0:
            feats = torch.stack(feats)
            main_labels = torch.stack(main_labels).unsqueeze(-1).long()
            
        return feats, main_labels