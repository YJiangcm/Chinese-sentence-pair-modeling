import os
import torch
import pandas as pd
from sys import platform
from torch.utils.data import DataLoader
import torch.nn as nn
from data import My_Dataset, load_embeddings
from utils import train, validate, test
from model import ESIM

def model_train_validate_test(train_df, dev_df, test_df, embeddings_file, vocab_file, target_dir, mode,
         num_labels=2,
         max_length=50,
         hidden_size=200,
         dropout=0.2,
         epochs=50,
         batch_size=256,
         lr=0.0005,
         patience=5,
         max_grad_norm=10.0,
         gpu_index=0,
         if_save_model=False,
         checkpoint=None):
    device = torch.device("cuda:{}".format(gpu_index) if torch.cuda.is_available() else "cpu")
    print(20 * "=", " Preparing for training ", 20 * "=")
    # 保存模型的路径
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # -------------------- Data loading ------------------- #
    print("\t* Loading training data...")
    train_data = My_Dataset(train_df, vocab_file, max_length, mode)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    print("\t* Loading validation data...")
    dev_data = My_Dataset(dev_df, vocab_file, max_length, mode)
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size)
    print("\t* Loading test data...")
    test_data = My_Dataset(test_df, vocab_file, max_length, mode)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    # -------------------- Model definition ------------------- #
    print("\t* Building model...")
    if(embeddings_file is not None):
        embeddings = load_embeddings(embeddings_file)
    else:
        embeddings = None
    model = ESIM(hidden_size, embeddings=embeddings, dropout=dropout, 
                 num_labels=num_labels, device=device).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    # -------------------- Preparation for training  ------------------- #
    criterion = nn.CrossEntropyLoss()
    # 过滤出需要梯度更新的参数
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer = optim.Adadelta(parameters, params["LEARNING_RATE"])
    optimizer = torch.optim.Adam(parameters, lr=lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", 
                                                           factor=0.85, patience=0)
    best_score = 0.0
    start_epoch = 1
    # Data for loss curves plot
    epochs_count = []
    train_losses = []
    valid_losses = []
    # Continuing training from a checkpoint if one was given as argument
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]
        print("\t* Training will continue on existing model from epoch {}...".format(start_epoch))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epochs_count = checkpoint["epochs_count"]
        train_losses = checkpoint["train_losses"]
        valid_losses = checkpoint["valid_losses"]
     # Compute loss and accuracy before starting (or resuming) training.
    _, valid_loss, valid_accuracy, _, = validate(model, dev_loader, criterion)
    print("\t* Validation loss before training: {:.4f}, accuracy: {:.4f}%".format(valid_loss, (valid_accuracy*100)))
    # -------------------- Training epochs ------------------- #
    print("\n", 20 * "=", "Training ESIM model on device: {}".format(device), 20 * "=")
    patience_counter = 0
    for epoch in range(start_epoch, epochs + 1):
        epochs_count.append(epoch)
        print("* Training epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = train(model, train_loader, optimizer,
                                                       criterion, epoch, max_grad_norm)
        train_losses.append(epoch_loss)
        print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%"
              .format(epoch_time, epoch_loss, (epoch_accuracy*100)))
        print("* Validation for epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy, _, = validate(model, dev_loader, criterion)
        valid_losses.append(epoch_loss)
        print("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%\n"
              .format(epoch_time, epoch_loss, (epoch_accuracy * 100)))
        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(epoch_accuracy)
        # Early stopping on validation accuracy.
        if epoch_accuracy < best_score:
            patience_counter += 1
        else:
            best_score = epoch_accuracy
            patience_counter = 0
            
            if (if_save_model):
                torch.save({"epoch": epoch, 
                        "model": model.state_dict(),
                        "best_score": best_score,
                        "epochs_count": epochs_count,
                        "train_losses": train_losses,
                        "valid_losses": valid_losses},
                        os.path.join(target_dir, "best.pth.tar"))
            
                print("save model succesfully!\n")
            
            print("* Test for epoch {}:".format(epoch))
            _, _, test_accuracy, predictions = validate(model, test_loader, criterion)
            print("Test accuracy: {:.4f}%\n".format(test_accuracy))
            test_prediction = pd.DataFrame({'prediction':predictions})
            test_prediction.to_csv(os.path.join(target_dir,"test_prediction.csv"), index=False)
        
        if patience_counter >= patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break


def model_load_test(test_df, vocab_file, embeddings_file, pretrained_file, 
                    test_prediction_dir, test_prediction_name, mode, num_labels, max_length=50, gpu_index=0, batch_size=128):

    device = torch.device("cuda:{}".format(gpu_index) if torch.cuda.is_available() else "cpu")
    print(20 * "=", " Preparing for testing ", 20 * "=")
    if platform == "linux" or platform == "linux2":
        checkpoint = torch.load(pretrained_file, map_location=device)
    else:
        checkpoint = torch.load(pretrained_file, map_location=device)
    # Retrieving model parameters from checkpoint.
    hidden_size = checkpoint["model"]["projection.0.weight"].size(0)
    num_classes = checkpoint["model"]["classification.6.weight"].size(0)
    embeddings = load_embeddings(embeddings_file)
    print("\t* Loading test data...")    
    test_data = My_Dataset(test_df, vocab_file, max_length, mode)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    print("\t* Building model...")
    model = ESIM(hidden_size, embeddings=embeddings, num_labels=num_labels, device=device).to(device)
    model.load_state_dict(checkpoint["model"])
    print(20 * "=", " Testing ESIM model on device: {} ".format(device), 20 * "=")
    batch_time, total_time, accuracy, predictions = test(model, test_loader)
    print("\n-> Average batch processing time: {:.4f}s, total test time: {:.4f}s, accuracy: {:.4f}%\n".format(batch_time, total_time, (accuracy*100)))
    test_prediction = pd.DataFrame({'prediction':predictions})
    if not os.path.exists(test_prediction_dir):
        os.makedirs(test_prediction_dir)
    test_prediction.to_csv(os.path.join(test_prediction_dir,test_prediction_name), index=False)
