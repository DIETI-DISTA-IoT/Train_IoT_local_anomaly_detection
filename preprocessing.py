import random
import torch
import numpy as np
from threading import Lock


def dict_to_tensor(data_dict):
    
    uncampled_values = [ (value  if isinstance(value, (int, float)) and not np.isnan(value) else 0.0) for value in data_dict.values() ]

    ## ALTERNATIVE 1: convert to numpy clamping the inf values:
    uncampled_values = np.array(uncampled_values, dtype=np.float32)
    clampled_values = np.clip(uncampled_values, a_min=-1e6, a_max=1e6)
    tensor = torch.tensor(clampled_values, dtype=torch.float32)

    ## ALTERNATIVE 2: convert to list clamping the inf values:
    # clampled_values = [ max(min(value, 30000), -4000) for value in uncampled_values ]
    # tensor = torch.tensor(clampled_values, dtype=torch.float32)

    ## ALTERNATIVE 3: convert to list clamping the inf values:
    # tensor = torch.tensor(uncampled_values, dtype=torch.float32)
    # tensor = torch.nan_to_num(tensor, posinf=1e6, neginf=-1e6)

    return tensor


class Buffer:
    def __init__(self, size, label=None, mode=None):
        self.size = size
        self.feats = []
        self.final_labels = []
        self.cluster_labels = []
        self.main_labels = []
        self.aux_labels = []
        self.label = label
        self.mode = mode
        self.lock = Lock()


    def add(self, feat_tensor, final_label_tensor, main_label_tensor, aux_label_tensor, cluster_label_tensor):
            
            with self.lock:
                self.feats.append(feat_tensor)
                self.final_labels.append(final_label_tensor)
                self.main_labels.append(main_label_tensor)
                self.aux_labels.append(aux_label_tensor)
                self.cluster_labels.append(cluster_label_tensor)
                if len(self.feats) > self.size:
                    self.feats.pop(0)
                    self.final_labels.pop(0)
                    self.main_labels.pop(0)
                    self.aux_labels.pop(0)
                    self.cluster_labels.pop(0)


    def format(self, item):

        final_label = None
        aux_label = None

        cluster_label = int(item.pop('cluster', None))
        cluster_label = torch.tensor(cluster_label, dtype=torch.long)
        main_label = torch.tensor(self.label, dtype=torch.long)

        if self.mode == 'SW':
            aux_label = 1 if item['node_status'] == 'INFECTED' else 0
            item.pop('node_status', None)
            final_label = self.transform_to_sereway_class_label(class_label=self.label, attack_label=aux_label)
            aux_label = torch.tensor(aux_label, dtype=torch.long)
            final_label = torch.tensor(final_label, dtype=torch.long)

        feat_tensor = dict_to_tensor(item)
        
        return feat_tensor, final_label, main_label, aux_label, cluster_label


    def transform_to_sereway_class_label(self, class_label, attack_label):
        """
        class, attack   -> sereway
        0,      0       -> 0
        0,      1       -> 1
        1,      0       -> 2
        1,      1       -> 3
        """
        sereway_label = (2*class_label) + attack_label
        return sereway_label
    

    def sample(self, n):
        
        feats = []
        final_labels = []
        main_labels = []
        aux_labels = []
        cluster_labels = []

        with self.lock:
            if len(self.feats) < n:
                feats = self.feats
                final_labels = self.final_labels
                main_labels = self.main_labels
                aux_labels = self.aux_labels
                cluster_labels = self.cluster_labels
            else:
                feats = random.sample(self.feats, n)
                final_labels = random.sample(self.final_labels, n)
                main_labels = random.sample(self.main_labels, n)
                aux_labels = random.sample(self.aux_labels, n)
                cluster_labels = random.sample(self.cluster_labels, n)

        if len(feats) > 0:
            feats = torch.stack(feats)
            main_labels = torch.stack(main_labels).unsqueeze(-1)
            cluster_labels = torch.stack(cluster_labels).unsqueeze(-1)

            if self.mode == 'SW':
                final_labels = torch.stack(final_labels).unsqueeze(-1)   
                aux_labels = torch.stack(aux_labels).unsqueeze(-1)
            

        return feats, final_labels, main_labels, aux_labels, cluster_labels