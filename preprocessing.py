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
        """
        self.buffer = []
        """
        self.feats = []
        self.labels = []
        self.cluster_labels = []

        self.label = label
        self.mode = mode
        self.lock = Lock()


    """
    def add(self, item):
        self.buffer.append(item)
        if len(self.buffer) > self.size:
            self.buffer.pop(0)
    """


    def add(self, feat_tensor, label_tensor, cluster_label_tensor):
            
            with self.lock:
                self.feats.append(feat_tensor)
                self.labels.append(label_tensor)
                self.cluster_labels.append(cluster_label_tensor)
                if len(self.feats) > self.size:
                    self.feats.pop(0)
                    self.labels.pop(0)
                    self.cluster_labels.pop(0)


    def format(self, item):

        cluster_label = int(item['cluster'])
        class_label = torch.tensor(self.label, dtype=torch.long)

        if self.mode == 'SW':
            attack_label = 1 if item['node_status'] == 'INFECTED' else 0
            sereway_label = self.transform_to_sereway_class_label(class_label=self.label, attack_label=attack_label)
            sereway_label = torch.tensor(sereway_label, dtype=torch.long)
            item.pop('node_status', None)
        
        item.pop('cluster', None)

        feat_tensor = dict_to_tensor(item)
        cluster_label = torch.tensor(cluster_label, dtype=torch.long)

        if self.mode == 'SW':
            return feat_tensor, sereway_label, cluster_label
        else:
            return feat_tensor, class_label, cluster_label


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

    """
    def sample(self, n):
        if len(self.buffer) < n:
            record_list = self.buffer
        else:
            record_list = random.sample(self.buffer, n)

        feature_tensors = []
        cluster_labels = []
        class_labels = []
        sereway_labels = []

        for record in record_list:
            cluster_label = int(record['cluster'])
            record_copy = record.copy()

            if self.mode == 'SW':
                attack_label = 1 if record['node_status'] == 'INFECTED' else 0
                sereway_label = self.transform_to_sereway_class_label(class_label=self.label, attack_label=attack_label)
                sereway_labels.append(sereway_label)
                record_copy.pop('node_status', None)
            
            record_copy.pop('cluster', None)
            feature_tensors.append(dict_to_tensor(record_copy))
            cluster_labels.append(cluster_label)
        
        if len(record_list) > 0:
            feature_tensors = torch.stack(feature_tensors)
            cluster_labels = torch.tensor(cluster_labels).unsqueeze(1)
            
            if self.mode == 'SW':
                class_labels = torch.tensor(sereway_labels).to(torch.float32)
            else:
                class_labels = torch.tensor([[self.label]] * len(feature_tensors)).to(torch.float32)



        return feature_tensors, class_labels, cluster_labels"
    """

    def sample(self, n):
        
        feats = []
        labels = []
        cluster_labels = []

        with self.lock:
            if len(self.feats) < n:
                feats = self.feats
                labels = self.labels
                cluster_labels = self.cluster_labels
            else:
                feats = random.sample(self.feats, n)
                labels = random.sample(self.labels, n)
                cluster_labels = random.sample(self.cluster_labels, n)

        if len(feats) > 0:
            feats = torch.stack(feats)
            labels = torch.stack(labels).unsqueeze(-1)
            cluster_labels = torch.stack(cluster_labels).unsqueeze(-1)

        return feats, labels, cluster_labels