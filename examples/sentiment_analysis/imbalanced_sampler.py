# 
import torch
import torch.utils.data
import torch.utils.data.sampler as sampler


class ImbalancedDatasetSampler(sampler.Sampler):

    def __init__(self, dataset, indices=None, num_samples=None):

        self.indices = list(range(len(dataset))) if indices is None else indices
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        label_to_count = {}
        label_to_count["true_negative"] = 0
        label_to_count["true_positive"] = 0

        for idx in self.indices:
            label = self.get_label(dataset, idx)
            label_to_count[label] += 1

        #print("TP and TN : {}".format(label_to_count))

        self.weights = torch.DoubleTensor([1.0/label_to_count[self.get_label(dataset, idx)] for idx in self.indices])

    def get_label(self, dataset, idx):
        if dataset.values[idx][2] == 1.0:
            return "true_positive"
        else:
            return "true_negative"

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples