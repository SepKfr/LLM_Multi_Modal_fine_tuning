import random
import numpy as np
import evaluate
import torch
from datasets import load_dataset
from torch import nn

from models.fine_tune_text_classifier import TextClassifierFineTune
from process_data.data_text_classification import TextClassification
from transformers import Adafactor
from transformers.optimization import AdafactorSchedule

torch.random.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = TextClassifierFineTune().to(device)

optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
lr_scheduler = AdafactorSchedule(optimizer)

imdb = load_dataset("imdb")
train_eval = imdb["train"].train_test_split(test_size=0.2)

text_cls_data = TextClassification(train=train_eval["train"], test=imdb["test"], val=train_eval["test"])

loss_fn = nn.CrossEntropyLoss()
epochs = 50
best_eval_loss = 1e10
check_p_epoch = 0
for epoch in range(epochs):
    tot_loss = 0
    model.train()
    for batch in text_cls_data.get_train_loader():

        inputs, labels = batch
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        tot_loss += loss.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    eval_loss = 0
    for batch in text_cls_data.get_val_loader():
        inputs, labels = batch
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        eval_loss += loss.item()

    print("train loss: {:.3f}".format(tot_loss))
    print("valid loss: {:.3f}".format(eval_loss))
    if eval_loss < best_eval_loss:
        best_eval_loss = eval_loss
        check_p_epoch = epoch
    if epoch - check_p_epoch >= 5:
        break


accuracy = evaluate.load("accuracy")
model.eval()
tot_acc = 0
for batch in text_cls_data.get_test_loader():
    inputs, labels = batch
    predicted = model(inputs)
    predicted = torch.argmax(predicted, dim=-1)
    acc = accuracy.compute(predictions=predicted, references=labels)
    tot_acc += acc['accuracy']

print("total accuracy: {:.3f}".format(tot_acc/len(text_cls_data.get_test_loader())))

