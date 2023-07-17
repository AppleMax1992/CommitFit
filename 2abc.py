
from transformers import AutoTokenizer, AutoModel, EarlyStoppingCallback, AutoModelForSequenceClassification, AutoConfig,Trainer, TrainingArguments,DataCollatorWithPadding
import torch
import numpy as np
import pandas as pd
# !pip install datasets
from datasets import load_metric


# %pip install evaluate
from evaluate import evaluator
from sklearn.model_selection import train_test_split
import csv
import re
csv.field_size_limit(500*1024*1024)

# df = pd.read_csv('D:\IET software-CC-CL\Commit Classification\experiment\Commit_dataset.csv', encoding="cp1252")

# label2id = {'a':'Adaptive','p':'Perfective','c':'Corrective'}
# df = df.replace({"3_labels": label2id})
# df = pd.read_csv(r'dataset.csv',engine="python")
# print(df['label'].value_counts())
# df['text'] = "['DIFF]" + df['diff']
# # df = df.replace({"2_labels": label2id})
# print(df)

train = pd.read_csv('train.csv',index_col=0)
train = train.rename(columns={'3_labels':'label','comment':'text'})
train.fillna(0, inplace=True)
test = pd.read_csv('test.csv',index_col=0)
test = test.rename(columns={'3_labels':'label','comment':'text'})
test.fillna(0, inplace=True)
print(len(test))






print(train)
from datasets import Dataset, load_metric
from commitfit.data  import (
    SAMPLE_SIZES,
    SEEDS,
    CommitFitDataset,
    create_fewshot_splits,
    create_fewshot_splits_multilabel,
    create_samples,
    get_templated_dataset,
    sample_dataset,
)


Dataset_train = Dataset.from_pandas(train)
test_dataset = Dataset.from_pandas(test)

train_dataset = get_templated_dataset(Dataset_train, candidate_labels=['positive','negative'], sample_size=8)



# test_shape = (len(test_dataset) , 90) 
# np_test = np.zeros(test_shape)

# filled_test = np.concatenate((_code_change, np_test), axis=0)


model_id = "sentence-transformers/all-mpnet-base-v2"

from commitfit import CommitFitModel
model = CommitFitModel.from_pretrained(model_id)


from commitfit import CommitFitTrainer
from sklearn.metrics import accuracy_score, recall_score, f1_score

def compute_metrics(y_pred, y_test):
    # print(y_pred,y_test)
    accuracy = accuracy_score(y_test,y_pred)
    # recall_score = accuracy_score(y_test,y_pred)
    # f1_score = f1_score(y_test,y_pred)

    # return {"accuracy": accuracy,"recall": recall_score, "f1":f1_score}
    return {"accuracy": accuracy}
trainer = CommitFitTrainer(
    model=model,
    train_dataset=train_dataset,
    train_code_change = train_code_change,
    test_code_change = test_code_change,
    eval_dataset=test_dataset,
    metric = compute_metrics,
    num_iterations=20,
    num_epochs=1
)
trainer.train(
    num_epochs=1, # The number of epochs to train the head or the whole model (body and head)
)
fewshot_metrics = trainer.evaluate()
print(fewshot_metrics)