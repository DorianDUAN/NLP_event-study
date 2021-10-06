# -*- coding:utf-8 -*-
# Author : @DrRic
# Version: 1.0
# Date : 2021/10/05 18:38
# Description: Solely for NUS BMF5342 Project 1 use

# bert feature base
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch import optim
import transformers as tfs
import math
import warnings

warnings.filterwarnings('ignore')

# data set
train_set = pd.read_csv('./train_set_1.csv', error_bad_lines=False, header=None)
test_set = pd.read_csv('./test_set.csv', error_bad_lines=False, header=None)
# print("Train set shape:", train_set.shape)


class BertClassificationModel(nn.Module):
    def __init__(self):
        super(BertClassificationModel, self).__init__()
        model_class, tokenizer_class, pretrained_weights = (tfs.BertModel, tfs.BertTokenizer, 'bert-base-uncased')
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.bert = model_class.from_pretrained(pretrained_weights)
        self.dense = nn.Linear(768, 3)  # Default number of hidden units in bert is 768, and the output unit is 3, which means three categories

    def forward(self, batch_sentences):
        batch_tokenized = self.tokenizer.batch_encode_plus(batch_sentences, add_special_tokens=True,
                                                           max_len=66,
                                                           pad_to_max_length=True)  # tokenize、add special token、pad
        input_ids = torch.tensor(batch_tokenized['input_ids'])
        attention_mask = torch.tensor(batch_tokenized['attention_mask'])
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        bert_cls_hidden_state = bert_output[0][:, 0, :]  # 提取[CLS]对应的隐藏状态
        linear_output = self.dense(bert_cls_hidden_state)
        return linear_output


train_inputs = train_set[1].values
train_targets = train_set[2].values
test_inputs = test_set[1].values
test_targets = test_set[2].values


batch_size = 64
batch_count = int(len(train_inputs) / batch_size)
batch_train_inputs, batch_train_targets = [], []
for i in range(batch_count):
    batch_train_inputs.append(train_inputs[i*batch_size: (i+1)*batch_size])
    batch_train_targets.append(train_targets[i*batch_size: (i+1)*batch_size])

# train the model
epochs = 5
lr = 0.01
print_every_batch = 5
bert_classifier_model = BertClassificationModel()
optimizer = optim.SGD(bert_classifier_model.parameters(), lr=lr, momentum=0.9)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    print_avg_loss = 0
    for i in range(batch_count):
        inputs = batch_train_inputs[i]
        labels = torch.tensor(batch_train_targets[i])
        optimizer.zero_grad()
        outputs = bert_classifier_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print_avg_loss += loss.item()
        if i % print_every_batch == (print_every_batch - 1):
            print("Batch: %d, Loss: %.4f" % ((i + 1), print_avg_loss / print_every_batch))
            print_avg_loss = 0


# save the model
torch.save(bert_classifier_model, 'bert_sentiment_model.pkl')


# eval the trained model
total = len(test_inputs)
hit = 0
with torch.no_grad():
    for i in range(total):
        outputs = bert_classifier_model([test_inputs[i]])
        _, predicted = torch.max(outputs, 1)
        if predicted == test_targets[i]:
            hit += 1

print("Accuracy: %.2f%%" % (hit / total * 100))




