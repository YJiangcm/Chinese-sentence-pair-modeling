# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 00:19:30 2020

@author: 31906
"""
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from transformers.optimization import AdamW
from sys import platform

from data import DataPrecessForSentence
from utils import train, validate, test
from models import Bert_wwm_Model

def model_train_validate_test(train_df, dev_df, test_df, target_dir, 
         max_seq_len=64,
         num_labels = 2,
         epochs=10,
         batch_size=32,
         lr=2e-05,
         patience=1,
         max_grad_norm=10.0,
         checkpoint=None):

    bertmodel = Bert_wwm_Model(requires_grad = True, num_labels = num_labels)
    tokenizer = bertmodel.tokenizer
    
    print(20 * "=", " Preparing for training ", 20 * "=")
    # 保存模型的路径，没有则创建文件夹
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # -------------------- Data loading ------------------- #
    print("\t* Loading training data...")
    train_data = DataPrecessForSentence(tokenizer, train_df, max_seq_len)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    print("\t* Loading validation data...")
    dev_data = DataPrecessForSentence(tokenizer,dev_df, max_seq_len)
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size)
    
    print("\t* Loading test data...")
    test_data = DataPrecessForSentence(tokenizer,test_df, max_seq_len) 
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    # -------------------- Model definition ------------------- #
    print("\t* Building model...")
    device = torch.device("cuda")
    model = bertmodel.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    # -------------------- Preparation for training  ------------------- #
    # 待优化的参数
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {
                    'params':[p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                    'weight_decay':0.01
            },
            {
                    'params':[p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                    'weight_decay':0.0
            }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    # 当网络的评价指标不在提升的时候，可以通过降低网络的学习率来提高网络性能
    # total_steps = len(train_loader) * epochs
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.85, patience=0)

    best_score = 0.0
    start_epoch = 1
    # Data for loss curves plot
    epochs_count = []
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []
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
        train_accuracy = checkpoint["train_accuracy"]
        valid_losses = checkpoint["valid_losses"]
        valid_accuracy = checkpoint["valid_accuracy"]
     # Compute loss and accuracy before starting (or resuming) training.
    _, valid_loss, valid_accuracy, _, = validate(model, dev_loader)
    print("\n* Validation loss before training: {:.4f}, accuracy: {:.4f}%".format(valid_loss, (valid_accuracy*100)))
    # -------------------- Training epochs ------------------- #
    print("\n", 20 * "=", "Training bert_wwm model on device: {}".format(device), 20 * "=")
    patience_counter = 0
    for epoch in range(start_epoch, epochs + 1):
        epochs_count.append(epoch)

        print("* Training epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = train(model, train_loader, optimizer, epoch, max_grad_norm)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%".format(epoch_time, epoch_loss, (epoch_accuracy*100)))
        
        print("* Validation for epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy, _, = validate(model, dev_loader)
        valid_losses.append(epoch_loss)
        valid_accuracies.append(epoch_accuracy)
        print("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%\n"
              .format(epoch_time, epoch_loss, (epoch_accuracy*100)))
        
        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(epoch_accuracy)
        # Early stopping on validation accuracy.
        if epoch_accuracy < best_score:
            patience_counter += 1
        else:
            best_score = epoch_accuracy
            patience_counter = 0
            torch.save({"epoch": epoch, 
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_score": best_score, # 验证集上的最优准确率
                        "epochs_count": epochs_count,
                        "train_losses": train_losses,
                        "train_accuracy": train_accuracies,
                        "valid_losses": valid_losses,
                        "valid_accuracy": valid_accuracies,
                        },
                        os.path.join(target_dir, "best.pth.tar"))
            print("save model succesfully!\n")
            
            print("* Test for epoch {}:".format(epoch))
            _, _, test_accuracy, predictions = validate(model, test_loader)
            print("Test accuracy: {:.4f}%\n".format(test_accuracy))
            test_prediction = pd.DataFrame({'prediction':predictions})
            test_prediction.to_csv(os.path.join(target_dir,"test_prediction.csv"), index=False)
             
        if patience_counter >= patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break


def model_load_test(test_df, target_dir, test_prediction_dir, test_prediction_name, num_labels=2, max_seq_len=64, batch_size=32):
    bertmodel = Bert_wwm_Model(requires_grad = False, num_labels = num_labels)
    tokenizer = bertmodel.tokenizer
    device = torch.device("cuda")
    print(20 * "=", " Preparing for testing ", 20 * "=")
    if platform == "linux" or platform == "linux2":
        checkpoint = torch.load(os.path.join(target_dir, "best.pth.tar"))
    else:
        checkpoint = torch.load(os.path.join(target_dir, "best.pth.tar"), map_location=device)
    # Retrieving model parameters from checkpoint.
    print("\t* Loading test data...")    

    test_data = DataPrecessForSentence(tokenizer,test_df, max_seq_len) 
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    print("\t* Building model...")
    
    model = bertmodel.to(device)
    model.load_state_dict(checkpoint["model"])
    print(20 * "=", " Testing BERT model on device: {} ".format(device), 20 * "=")
    batch_time, total_time, accuracy, predictions = test(model, test_loader)
    print("\n-> Average batch processing time: {:.4f}s, total test time: {:.4f}s, accuracy: {:.4f}%\n".format(batch_time, total_time, (accuracy*100)))
    test_prediction = pd.DataFrame({'prediction':predictions})
    if not os.path.exists(test_prediction_dir):
        os.makedirs(test_prediction_dir)
    test_prediction.to_csv(os.path.join(test_prediction_dir,test_prediction_name), index=False)


if __name__ == "__main__":
    lcqmc_path = "/content/drive/My Drive/Sentence_pair_modeling/LCQMC/"
    train_df = pd.read_csv(os.path.join(lcqmc_path, "data/train.tsv"),sep='\t',header=None, names=['s1','s2','label'])
    dev_df = pd.read_csv(os.path.join(lcqmc_path, "data/dev.tsv"),sep='\t',header=None, names=['s1','s2','label'])
    test_df = pd.read_csv(os.path.join(lcqmc_path, "data/test.tsv"),sep='\t',header=None, names=['s1','s2','label'])
    target_dir = os.path.join(lcqmc_path, "output/Bert_wwm/")
    model_load_test(test_df, target_dir, test_prediction_dir)