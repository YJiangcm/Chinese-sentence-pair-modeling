# Chinese-sentence-pair-modeling

This repository contains the following models for sentence pair modeling: **BiLSTM (max-pooling), BiGRU (element-wise product), BiLSTM (self-attention), ABCNN, RE2, ESIM, BiMPM, Siamese BERT, BERT, RoBERTa, XLNet, DistilBERT and ALBERT.** All these codes are based on PyTorch and you are recommended to run the "ipynb" files in Google Colab, where you could get GPU resources for free.

## 1. Datasets
I conduct experiments on 5 Chinese datasets: 3 paraphrase identification datasets and 2 natural language inference datasets. Tables below give a brief comparison of these datasets.

<img src="https://github.com/YJiangcm/Chinese-sentence-pair-modeling/blob/master/photos/datasets.PNG" width="600" height="450">

**Note**: the BQ Corpus dataset requires you to send an application form, which can be downloaded from http://icrc.hitsz.edu.cn/Article/show/175.html .
The CMNLI dataset is too large, and you can download it from https://storage.googleapis.com/cluebenchmark/tasks/cmnli_public.zip . Due to the unbalanced categories (labels of 1, 2, 3, 4 account for a smallpercentage) of ChineseSTS, I drop these few labels and converts the dataset into an binary classification task. What's more, OCNLI and CMNLI datasets are preprocessed by removing the data with missing labels.

## 2. Implementation Details
After analyzing the distributions of lengths of sentences in 5 datasets, the max\_sequence\_length for truncation is set to 64 for convenient comparisons. What's more, the hidden size is set to 200 in all models using BiLSTM.

For models of BiLSTM (max-pooling), BiGRU (element-wise product), BiLSTM (self-attention), ABCNN, RE2, ESIM and BiMPM, I apply character embedding and word embedding respectively while tokenizing sentences into tokens.  The pre-trained character embedding matrix contains 300-dimensional character vectors trained on Wikipedia\_zh corpus (please download it from https://github.com/liuhuanyong/ChineseEmbedding/blob/master/model/token_vec_300.bin), while the word embedding matrix is composed of 300-dimensional word vectors trained on Baidu Encyclopedia (please download it from https://pan.baidu.com/s/1Rn7LtTH0n7SHyHPfjRHbkg). 

As for Siamese BERT, BERT, BERT-wwm, RoBERTa, XLNet DistilBERT and ALBERT, learning rate is the most important hyperparameter (inappropriate choice may lead to divergence of models), which is generally chosen from 1e-5 to 1e-4. What's more, it should also be determined by the batchsize. A large batchsize should correspond to a large learning rate.

## 3. Experiment results and analysis

The following table shows the test accuracy (%) of different models on 5 datasets:


|          Model          | LCQMC | ChineseSTS | BQ Corpus | OCNLI | CMNLI |
|:-----------------------:|:-----:|:----------:|:---------:|:-----:|:-----:|
|  BiLSTM (max-pooling)-char-pre  |  74.4 |    97.5    |    70.0   |  60.6 |  56.7 |
|  BiLSTM (max-pooling)-word-pre  |  75.2 |    98.0    |    68.0   |  58.0 |  56.9 |
| BiLSTM (self-attention)-char-pre |  85.0 |    96.8    |    79.8   |  58.5 |  63.6 |
| BiLSTM (self-attention)-word-pre |  83.7 |    94.4    |    79.3   |  57.8 |  64.2 |
|      ABCNN-char-pre     |  79.5 |    97.2    |    78.8   |  53.2 |  63.2 |
|      ABCNN-word-pre     |  81.3 |    97.9    |    74.4   |  54.1 |  59.8 |
|       RE2-char-pre      |  84.2 |    98.7    |    80.4   |  61.0 |  68.6 |
|       RE2-word-pre      |  84.5 |    98.6    |    80.1   |  57.2 |  65.1 |
|      ESIM-char-pre      |  83.6 |    99.0    |    81.2   |  64.8 |  74.0 |
|      ESIM-word-pre      |   84  |    98.9    |    81.7   |  61.3 |  72.6 |
|      BiMPM-char-pre     |  83.6 |    98.9    |    79.2   |  63.9 |  69.7 |
|      BiMPM-word-pre     |  83.7 |    98.8    |    80.3   |  59.9 |  69.6 |
|       Siamese BERT      |  84.8 |    97.7    |    83.5   |  66.8 |  72.5 |
|           BERT          |  **87.8** |    98.9    |    84.2   |  73.8 |  80.5 |
|         BERT-wwm        |  87.4 |    99.2    |    84.5   |  73.8 |  80.6 |
|         RoBERTa         |  87.5 |    99.2    |    **84.6**   |  **75.5** |  80.6 |
|          XLNet          |  87.4 |    99.1    |    84.1   |  73.6 |  **80.7** |
|          ALBERT         |  87.4 |    **99.5**    |    82.2   |  68.1 |  74.8 |

### 3.1 Char Embedding vs. Word Embedding
<img src="https://github.com/YJiangcm/Chinese-sentence-pair-modeling/blob/master/photos/char_vs_word.png" width="600" height="450">

Note that the y_axis is the averaged accuracy on 5 different test sets. We can see that using method of char embedding gets greater performance than that of word embedding. It may be because that the word embedding matrix is much more sparse than char embedding matrix, so large quantities of weights of word vectors do not get updated during training. Besides, the out-of-vocabulary problem is more easily to happen in word embedding, which also weakens its performance. 

### 3.2 Comparison of Average Test Accuracy on 5 Datasets
<img src="https://github.com/YJiangcm/Chinese-sentence-pair-modeling/blob/master/photos/accuracy.png" width="600" height="450">
Here character embedding is chosen for BiLSTM (max-pooling), BiLSTM (self-attention), ABCNN, RE2, ESIM and BiMPM, and the accuracy is computed by taking average on 5 datasets. We can see that RoBERTa model gets the best performance among these models, and BERT-wwm is slightly better than BERT.

### 3.3 Comprehensive Evaluation of the Models 

(P.S. the original papers can be accessed by clicking the hyperlinks)

|          Model          | Accuracy\(%) | Number of parameters  (millions) | Average training time  (seconds / sentence pair)  | Average inference time  (seconds / sentence pair) |
|:-----------------------:|:------------:|:-------------------------------:|:-------------------------------------------------:|:-------------------------------------------------:|
|   [BiLSTM (max-pooling)](https://arxiv.org/pdf/1705.02364.pdf)  |     71.8     |                16               |                      7.4E-04                      |                      1.6E-04                      |
| [BiLSTM (self-attention)](https://arxiv.org/pdf/1705.02364.pdf) |     76.7     |                16               |                      7.5E-04                      |                      1.7E-04                      |
|       [Siamese BERT](https://arxiv.org/pdf/1908.10084.pdf)      |     81.1     |               102               |                      1.5E-02                      |                      3.9E-03                      |
|          [ABCNN](https://arxiv.org/pdf/1512.05193.pdf)          |     74.4     |                13               |                      **4.8E-04**                      |                      1.3E-04                      |
|           [RE2](https://arxiv.org/pdf/1908.00300.pdf)           |     78.6     |                16               |                      8.1E-04                      |                      2.1E-04                      |
|           [ESIM](https://arxiv.org/pdf/1609.06038.pdf)          |     80.5     |                17               |                      5.5E-04                      |                      **1.2E-04**                      |
|          [BiMPM](https://arxiv.org/pdf/1702.03814.pdf)          |     79.1     |                13               |                      2.0E-03                      |                      9.1E-04                      |
|           [BERT](https://arxiv.org/pdf/1810.04805.pdf)         |     85.0     |               102               |                      6.7E-03                      |                      2.1E-03                      |
|         [BERT-wwm](https://arxiv.org/pdf/1906.08101.pdf)        |     85.1     |               102               |                      6.8E-03                      |                      2.1E-03                      |
|         [RoBERTa](https://arxiv.org/pdf/1907.11692.pdf)         |     **85.5**     |               102               |                      1.1E-02                      |                      3.7E-03                      |
|          [XLNet](https://arxiv.org/pdf/1906.08237.pdf)          |     85.0     |               117               |                      9.5E-03                      |                      3.6E-03                      |
|          [ALBERT](https://arxiv.org/pdf/1909.11942.pdf)         |     82.4     |                **12**               |                      1.1E-02                      |                      3.7E-03                      |


## LICENSE
These codes are mainly inspired by https://github.com/zhaogaofeng611/TextMatch. As for this repository, please refer to [Apache-2.0 License Copyright (c) 2020 YJiangcm](https://github.com/YJiangcm/Chinese-sentence-pair-modeling/blob/master/LICENSE).
