from SentenceEvaluator import SentenceEvaluator
import torch
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
from util import batch_to_device
import os
import csv
from itertools import chain

class LabelAccuracyEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on its accuracy on a labeled dataset

    This requires a model with LossFunction.SOFTMAX

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """

    def __init__(self, dataloader: DataLoader, name: str = "", softmax_model = None):
        """
        Constructs an evaluator for the given dataset

        :param dataloader:
            the data for the evaluation
        """
        self.dataloader = dataloader
        self.name = name
        self.softmax_model = softmax_model

        if name:
            name = "_"+name

        self.csv_file = "test_"+name+"accuracy.csv"
        self.csv_headers = ["epoch", "steps", "accuracy"]
        
        self.csv_file_result = "test_"+name+"prediction.csv"
        self.csv_headers_result = ["prob_0", "prob_1", "prediction"]

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        model.eval()
        total = 0
        correct = 0

        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logging.info("Evaluation on the "+self.name+" dataset"+out_txt)
        self.dataloader.collate_fn = model.smart_batching_collate
        
        prob_0 = []
        prob_1 = []
        predicts=[]
        for step, batch in enumerate(tqdm(self.dataloader, desc="Evaluating")): #遍历整个数据集
            features, label_ids = batch_to_device(batch, model.device)
            with torch.no_grad():
                _, prediction = self.softmax_model(features, labels=None)

            total += prediction.size(0)
            correct += torch.argmax(prediction, dim=1).eq(label_ids).sum().item()
            prob_0.append(prediction.permute(1,0)[0].cpu().numpy().tolist())
            prob_1.append(prediction.permute(1,0)[1].cpu().numpy().tolist())
            predicts.append(torch.argmax(prediction, dim=1).cpu().numpy().tolist())
        prob_0 = list(chain.from_iterable(prob_0))
        prob_1 = list(chain.from_iterable(prob_1))
        predicts = list(chain.from_iterable(predicts))
        accuracy = correct/total

        logging.info("Accuracy: {:.4f} ({}/{})\n".format(accuracy, correct, total))

        if output_path is not None:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, accuracy])
            else:
                with open(csv_path, mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, accuracy])

            csv_path1 = os.path.join(output_path, self.csv_file_result)
            
            if not os.path.isfile(csv_path1):
                with open(csv_path1, mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers_result)
                    for row in zip(prob_0, prob_1, predicts):
                        writer.writerow([row[0], row[1], row[2]])
            else:
                with open(csv_path1, mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers_result)
                    for row in zip(prob_0, prob_1, predicts):
                        writer.writerow([row[0], row[1], row[2]])
                        
        return accuracy