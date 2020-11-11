from transformers import BertTokenizer, BertModel
import torch
import json
import os
import sys
sys.append('/drives/sdf/preprocess/utils/')

from utils.helpers import find_project_root

input_txt = ['Hello World', 'Bye bye, bye bye']
tokenizer = BertTokenizer.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2')
model = BertModel.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2')
batch = tokenizer.batch_encode_plus(input_txt, return_tensors='pt', padding=True)
outputs = model(**batch)
last_hidden_states = outputs[0]
logger.info(last_hidden_states)
__import__('pdb').set_trace()


# another method
from transformers import pipeline

pipe = pipeline(task='feature-extraction', model='digitalepidemiologylab/covid-twitter-bert-v2')
res = pipe(input_txt)



# # pretrain (usually done for you)
# * unsupervised
# * data consists only of raw text
# * we are "creating" a supervised problem from raw data -> NSP/MLM tasks (Multi-task learning)
# * The final pretrained model is a "double head model" (one for NSP and one for MLM)
#
# # finetune
# We typically collect some form of annotation data, of the form text -> label/class (usually much smaller dataset than pretrain)
# We remove the two heads from the pretrain model (NSP/MLM) and replace it with a new head (e.g. a classification head)
# tokenized input ids (tweet) -> transformer (encoder) (batch size x seq length x hidden size, aka last hidden state)  -> classsification head (batch size x num classes)
# what is a classification head?
# it is a single neural layer which maps the hidden state to the output logits (num classes). The weights are initialized randomly.
# in finetuning we initialize the encoder weights from BERT-large and randomly initialize the classification head. Then we use a loss function (typically sparse entropy loss)
# we use an optimizer (typically ADAM), and we train *all* weights in the model. The idea is that the BERT weights do not drastically change so we want to use a small learning rate
# (typically we use slanted triangular learning schedule, meaning warmup of 10k steps, and then linear decay).

# CT-BERT
# We used another! technique called DSP (Domain Specific Pretraining). 
# * Start with BERT-Large-uncased
# * Do additional pretraining on our Twitter corpus
# * Eventually we evaluate (using finetuning) on annotation datasets from SemEval competitions


