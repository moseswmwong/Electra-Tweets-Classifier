# Electra-Tweets-Classifier

## Overview

This is a NLP Kaggle Challenge I attend in May 2021, by using the state of the art language model, it is able to achieve 1.0 (full) Kaggle score, and ranked top 5% in the competition.

This is a classification problem, the objective is to identify emergency tweet at high accuracy. Kaggle is providing about 4500 tweets each label as 1 meaning emergency, or 0 meaning non-emergency. The challenge is to correctly predict whether a unseen example is emergency tweet or not.

## Problem Analysis

Maximum length of tweet is 280 characters long, and since leading Transformer model e.g. BERT support Tranformer mechanism with 512 maximum input token, therefore the model can well suitable to seek attention to the full length of the tweet text even when the language model is processing at the last token of input. In conclusion, BERT or the more advanced Electra are both able to capture attention support at any given input token the full length of the tweet. This is a good news to this problem because many language model and corresponding problems, e.g. long text summarization, suffer from diminishing effect due to long input text, which is not a problem here.

All input text are English so the English pre-trained model BERT, or Electra are both suitable to solve this problem.

Transformer is selected in this problem because the Transformer language model is able to capture strong co-relationship between words in the input text, and word embedding used at the beginning of the text processing further enhance the ability for the model to capture meaning of the input text while furthering these important information downstream during training, in particular, weight formation inside multi-head attention mechanism.

## Alternatives

Bag-of-word method was the closest canadidate before Transformer was selected to solve this problem, and after gone through the estimated performance of Bag-of-word, there are a few major disadvantages, first, does not capture the meaning of each word, second, it does not consider sequence nor reasoning nor use of language-structure with the input text, and finally, it can misinterpret short phases for example, the phase "not good" and "good" are counted as one "not", and two "good" which bag-of-word consider a positive signal which in reality their effect are opposite and cancelling out each other.

Transformer is the best option because it is the state-of-the-art language model that before it make the decision it is able to capture all meanings from the whole input text, understand the meaning of each word, and learn to consider inter-relationship between words in the input text. Last but not least, not only it consider inter-relationship of a word during processing with other words before it, the bidirectional nature of the BERT / Electra models consider words after too.

BERT is a pre-trained bidirectional Transformer model that can handle the task well, however Electra do a even better job by enhancing the MLM (20% token omission) pre-training phase of BERT, making it a more accurate language model.

## Solution

This program use Electra base pre-trained model provided by Google fine-tuning by training 15 epochs using the Kaggle provided emergency tweet text, and use the model to predict unseen test dataset provided by the competition and the result is 1.0 Kaggle score.

## Points to note

- Python 3.7 or above
- The program use PyTorch
- GPU is required. It was trained on a NVIDIA Telsa P100 GPU and total program execution time (including training) is less than 30 minutes
- You need to download dataset from Kaggle before running the program, the link is here - https://www.kaggle.com/c/nlp-getting-started


Feel free to download and run it, enjoy!

