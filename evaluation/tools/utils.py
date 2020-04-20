import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Meter:
    """
     This class is used to keep track of the metrics in the train and dev loops.
    """
    def __init__(self, target_classes):
        """
        :param target_classes: The classes for whom the metrics will be calculated.
        """
        self.target_classes = target_classes

        self.loss = 0

        self.micro_prec = 0
        self.micro_recall = 0
        self.micro_f1 = 0

        self.macro_prec = 0
        self.macro_recall = 0
        self.macro_f1 = 0

        self.it = 0

    def update_params(self, loss, logits, y_true):
        """
        Update the metrics.

        :param loss: The current loss.
        :param logits: The current logits.
        :param y_true: The current true labels.
        :return:
        """
        # get the argmax of logits from each output
        y_pred = torch.tensor([torch.argmax(x) for x in logits.view(-1, logits.shape[2])]).tolist()
        y_true = y_true.reshape(-1).tolist()

        for i in range(len(y_pred)):
            y_pred[i] = y_true[i] if y_true[i] not in self.target_classes else y_pred[i]

        # print(y_pred)
        # print(y_true)
        # print(self.target_classes)

        # compute the micro precision/recall/f1, macro precision/recall/f1
        micro_prec = precision_score(y_true, y_pred, labels=self.target_classes, average='micro', zero_division=1)
        micro_recall = recall_score(y_true, y_pred, labels=self.target_classes, average='micro', zero_division=1)
        micro_f1 = f1_score(y_true, y_pred, labels=self.target_classes, average='micro', zero_division=1)

        macro_prec = precision_score(y_true, y_pred, labels=self.target_classes, average='macro', zero_division=1)
        macro_recall = recall_score(y_true, y_pred, labels=self.target_classes, average='macro', zero_division=1)
        macro_f1 = f1_score(y_true, y_pred, labels=self.target_classes, average='macro', zero_division=1)

        self.loss = (self.loss * self.it + loss) / (self.it + 1)

        self.micro_prec = (self.micro_prec * self.it + micro_prec) / (self.it + 1)
        self.micro_recall = (self.micro_recall * self.it + micro_recall) / (self.it + 1)
        self.micro_f1 = (self.micro_f1 * self.it + micro_f1) / (self.it + 1)

        self.macro_prec = (self.macro_prec * self.it + macro_prec) / (self.it + 1)
        self.macro_recall = (self.macro_recall * self.it + macro_recall) / (self.it + 1)
        self.macro_f1 = (self.macro_f1 * self.it + macro_f1) / (self.it + 1)

        self.it += 1

        return self.loss, \
               self.micro_prec, self.micro_recall, self.micro_f1, \
               self.macro_prec, self.macro_recall, self.macro_f1

    def reset(self):
        """
        Resets the metrics to the 0 values. Must be used after each epoch.
        """
        self.loss = 0

        self.micro_prec = 0
        self.micro_recall = 0
        self.micro_f1 = 0

        self.macro_prec = 0
        self.macro_recall = 0
        self.macro_f1 = 0

        self.it = 0


def print_info(target_classes, weights, label_encoder, lang_model_name, fine_tune, device):
    print("Training session info:")
    print("\tLanguage model: {}, Finetune: {}".format(lang_model_name, fine_tune))
    print("\tTarget classes: {}".format([label_encoder.inverse_transform([target_class])[0] for target_class in target_classes]))
    print("\tAll classes: {}".format(label_encoder.classes_.tolist()))
    print("\tLabel Weights: {}".format(weights.tolist()))
    print("\tDevice: {}".format(device))
