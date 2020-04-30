from transformers import *
import torch
from sklearn.preprocessing import LabelEncoder


def load_data_from_file(path,
                        batch_size,
                        tokens_column, predict_column,
                        lang_model,
                        max_len,
                        separator,
                        pad_label, null_label,
                        device,
                        label_encoder=None,
                        shuffle=True):
    # create the tokenizer for subtokens
    tokenizer = AutoTokenizer.from_pretrained(lang_model,
                                              do_lower_case=True if "uncased" in lang_model else False)
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id

    list_all_tokens = []
    list_all_labels = []
    list_all_masks = []

    with open(path, "r", encoding='utf-8') as file:
        list_tokens = []
        list_labels = []

        for line in file:
            if not line.startswith("#"):
                if line != "\n":
                    tokens = line.split(separator)

                    token = tokens[tokens_column]
                    label = tokens[predict_column].replace("\n", "")

                    # subtokenize each token
                    subtokens = tokenizer.encode(token, add_special_tokens=False)

                    # add the subtokens to the list of tokens
                    list_tokens += subtokens
                    # only the first subtoken retains the token value, the rest are marked with the null label - <X>
                    list_labels += [label] + [null_label] * (len(subtokens) - 1)
                else:
                    assert len(list_tokens) == len(list_labels)

                    if len(list_tokens) == 0:
                        continue

                    if len(list_tokens) + 2 <= max_len:
                        # add the tokens to a collective list, append [CLS] token to beginning an the [SEP] token to the end
                        list_all_tokens.append(torch.tensor([cls_token_id] + list_tokens + [sep_token_id]))
                        # add the labels to a collective list, append <pad> to the beginning and the end
                        list_all_labels.append([pad_label] + list_labels + [pad_label])
                        # create the mask - ignore all tokens from [SEP] token onwards
                        list_all_masks.append(torch.tensor([1] * (len(list_tokens) + 2)))

                    assert len(list_tokens) == len(list_labels)

                    list_tokens = []
                    list_labels = []

    assert len(list_all_tokens) == len(list_all_labels) == len(list_all_masks)

    # fit the label encoder
    if label_encoder is None:
        label_encoder = LabelEncoder()
        label_encoder.fit(sum(list_all_labels, []))

    # encode the labels -> transform strings in integers that represent the classes
    list_all_encoded_labels = [torch.tensor(label_encoder.transform([label if label in label_encoder.classes_ else
                                                                              null_label for label in list_labels]))
                                                                     for list_labels in list_all_labels]

    # pad the tokens, the labels and the masks
    X = torch.nn.utils.rnn.pad_sequence(list_all_tokens, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
    y = torch.nn.utils.rnn.pad_sequence(list_all_encoded_labels, batch_first=True, padding_value=label_encoder.transform([pad_label])[0]).to(device)
    masks = torch.nn.utils.rnn.pad_sequence(list_all_masks, batch_first=True, padding_value=0).to(device)

    # create the loader
    dataset = torch.utils.data.TensorDataset(X, y, masks)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader, label_encoder


def load_data(train_path, dev_path,
              batch_size,
              tokens_column, predict_column,
              lang_model, max_len,
              separator,
              pad_label, null_label,
              device):
    """
    Function that loads the training and the development data.

    :param train_path: Path to the training data.
    :param dev_path: Path to the development data.
    :param batch_size: The batch size.
    :param tokens_column: The column of the tokens in the data file.
    :param predict_column: The column that must be predicted.
    :param lang_model: The name of the language model that will be used. See HuggingFace's available models.
    :param max_len: The maximum length of the sequences.
    :param separator: The separator of the data files.
    :param pad_label: The padding label that will be used. Defaults to <pad>.
    :param null_label: The null label that will be used to mask the second to last subtokens's label. Defaults to <X>.
    :param device: The device used for training.
    :return: Two pytorch loaders for train and dev, and the label encoder (from scikit) used to encode the labels.
    """
    train_loader, label_encoder = load_data_from_file(train_path,
                                                      batch_size,
                                                      tokens_column, predict_column,
                                                      lang_model,
                                                      max_len,
                                                      separator,
                                                      pad_label, null_label,
                                                      device)

    dev_loader, _ = load_data_from_file(dev_path,
                                        batch_size,
                                        tokens_column, predict_column,
                                        lang_model,
                                        max_len,
                                        separator,
                                        pad_label, null_label,
                                        device,
                                        label_encoder)

    return train_loader, dev_loader, label_encoder
