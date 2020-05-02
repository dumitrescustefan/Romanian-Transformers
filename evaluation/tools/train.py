import argparse, os, pickle, torch
from load import load_data
from model import LangModelWithDense
from tqdm.autonotebook import tqdm as tqdm
from transformers import *
from utils import Meter



def train_model(model,
                train_loader, dev_loader,
                optimizer, criterion,
                num_classes, target_classes, it,
                label_encoder,
                device):

    # create to Meter's classes to track the performance of the model during training and evaluating
    train_meter = Meter(target_classes)
    dev_meter = Meter(target_classes)

    best_f1 = 0
    loss, macro_f1 = 0, 0

    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    curr_patience = 0

    # epoch loop
    for epoch in range(args.epochs):
        train_tqdm = tqdm(train_loader, leave=False)

        model.train()

        # train loop
        for i, (train_x, train_y, mask) in enumerate(train_tqdm):
            train_tqdm.set_description(
                "    Training - Epoch: {}/{}, Loss: {:.4f}, F1: {:.4f}, Best F1: {:.4f}".
                format(epoch + 1, args.epochs, loss, macro_f1, best_f1))
            train_tqdm.refresh()

            # get the logits and update the gradients
            optimizer.zero_grad()

            logits = model.forward(train_x, mask)

            loss = criterion(logits.reshape(-1, num_classes).to(device), train_y.reshape(-1).to(device))
            loss.backward()
            optimizer.step()

            if args.fine_tune:
                scheduler.step()

            # get the current metrics (average over all the train)
            loss, _, _, _, _, _, macro_f1 = train_meter.update_params(loss.item(), logits, train_y)

        # reset the metrics to 0
        train_meter.reset()

        dev_tqdm = tqdm(dev_loader, leave=False)
        model.eval()
        loss, macro_f1 = 0, 0

        # evaluation loop -> mostly same as the training loop, but without updating the parameters
        for i, (dev_x, dev_y, mask) in enumerate(dev_tqdm):
            dev_tqdm.set_description(
                "    Evaluating - Epoch: {}/{}, Loss: {:.4f}, F1: {:.4f}, Best F1: {:.4f}".
                format(epoch + 1, args.epochs, loss, macro_f1, best_f1))
            dev_tqdm.refresh()

            logits = model.forward(dev_x, mask)
            loss = criterion(logits.reshape(-1, num_classes).to(device), dev_y.reshape(-1).to(device))

            loss, _, _, micro_f1, _, _, macro_f1 = dev_meter.update_params(loss.item(), logits, dev_y)

        dev_meter.reset()

        # if the current macro F1 score is the best one -> save the model
        if macro_f1 > best_f1:
            curr_patience = 0
            best_f1 = macro_f1
            torch.save(model, os.path.join(args.save_path, "model_{}.pt".format(it + 1)))
            with open(os.path.join(args.save_path, "label_encoder.pk"), "wb") as file:
                pickle.dump(label_encoder, file)
        else:
            curr_patience += 1

        if curr_patience > args.patience:
            break

    return best_f1


def train(train_loader, dev_loader, label_encoder, device):
    it_tqdm = tqdm(range(args.iterations))
    results = []

    AutoModel.from_pretrained(args.lang_model_name)

    it_tqdm.set_description("Iteration: {}/{}, Results F1: {}, Mean F1: {:.4f}"
                            .format(1, args.iterations, [], 0))
    it_tqdm.refresh()

    for it in it_tqdm:
        # select the desired language model and get the embeddings size
        lang_model = AutoModel.from_pretrained(args.lang_model_name)
        input_size = 768 if "base" in args.lang_model_name else 1024

        # create the model, the optimizer (weights are set to 0 for <pad> and <X>) and the loss function
        model = LangModelWithDense(lang_model, input_size, len(label_encoder.classes_), args.fine_tune).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        weights = torch.tensor(
            [1 if label != args.pad_label and label != args.null_label else 0 for label in label_encoder.classes_],
            dtype=torch.float32).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)

        # remove the null_label (X), the pad label (<pad>) and the (O)-for NER only from the evaluated targets during training
        classes = label_encoder.classes_.tolist()
        classes.remove(args.null_label)
        classes.remove(args.pad_label)
        if args.remove_o_label:
            classes.remove("O")
        target_classes = [label_encoder.transform([clss])[0] for clss in classes]

        # start training
        best_f1 = train_model(model,
                              train_loader, dev_loader,
                              optimizer, criterion,
                              len(label_encoder.classes_), target_classes, it,
                              label_encoder,
                              device)

        results.append(best_f1)

        it_tqdm.set_description("Iteration: {}/{}, Results F1: {}, Mean F1: {:.4f}"
                                .format(it + 1, args.iterations,
                                        ["{:.4f}".format(elem) for elem in results],
                                        0 if len(results) == 0 else sum(results) / (it + 1)))
        it_tqdm.refresh()


def main():
    device = torch.device(args.device)

    # Loading the train and dev data and save them in a loader + the encoder of the classes
    train_loader, dev_loader, label_encoder = load_data(args.train_path,
                                                        args.dev_path,
                                                        args.batch_size,
                                                        args.tokens_column, args.predict_column,
                                                        args.lang_model_name,
                                                        args.max_len,
                                                        args.separator,
                                                        args.pad_label,
                                                        args.null_label,
                                                        device)

    train(train_loader, dev_loader, label_encoder, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_path", type=str)
    parser.add_argument("dev_path", type=str)
    parser.add_argument("predict_column", type=int)
    parser.add_argument("--tokens_column", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--save_path", type=str, default="models")
    parser.add_argument("--lang_model_name", type=str, default="bert-base-cased")
    parser.add_argument("--fine_tune", action="store_true")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--separator", type=str, default="\t")
    parser.add_argument("--pad_label", type=str, default="<pad>")
    parser.add_argument("--null_label", type=str, default="<X>")
    parser.add_argument("--remove_o_label", action="store_true")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    main()