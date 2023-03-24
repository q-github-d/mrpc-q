
from __future__ import print_function

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import pandas as pd
import string
import nltk
import pickle
nltk.download('stopwords')


MAX_SEQUENCE_LENGTH = 25



def get_list(reader):
  sentence1 = []
  sentence2 = []
  totalrows = 0
  for ind, row in reader.iterrows():
    totalrows+=1
    if row['sentence1'] is None:
      continue
    if row['sentence2'] is None:
      continue
    sentence1.append(row['sentence1'])
    sentence2.append(row['sentence2'])
  print(totalrows)
  print('Question pairs: %d' % len(sentence1))
  return sentence1, sentence2

def remove_punctuation(list_of_test):
  final_list = []
  stopwords = nltk.corpus.stopwords.words('english')
  for x in list_of_test:
    punctuationfree="".join([i.lower() for i in x if i not in string.punctuation])
    filtered = ' '.join([word for word in punctuationfree.split() if word not in stopwords])
    final_list.append(filtered)
  return final_list


def my_tokenizer(sentence1, sentence2):
  with open('tokenizer.pickle', 'rb') as handle:
    tokenizer2 = pickle.load(handle)
  sentence1_word_sequences = tokenizer2.texts_to_sequences(sentence1)
  sentence2_word_sequences = tokenizer2.texts_to_sequences(sentence2)
  return sentence1_word_sequences, sentence2_word_sequences


def finaL_save(sentence1_word_sequences, sentence2_word_sequences):
  q1_data = pad_sequences(sentence1_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
  q2_data = pad_sequences(sentence2_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
  print('Shape of sentence1 data tensor:', q1_data.shape)
  print('Shape of sentence2 data tensor:', q2_data.shape)
  return q1_data, q2_data
