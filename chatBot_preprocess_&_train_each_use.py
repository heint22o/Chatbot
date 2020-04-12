
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tensorflow
import tflearn
import random
import json

with open('intents.json') as file:
  data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])
#word stemming so that the "happening" will be the same as "happen"
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

#represent each sentence with a list the length of words in the models vocab
#each position will represent a word in the vocab
#if position in the list is a 1 then the word is in the sentence
#if position is a 0 then the word is not in the sentence
training = []

#create output lists which are the length of the amount of labels/tags in the dataset
#each position in the list will represent a distinct label/tag
#a 1 in any of those positions will show which label/tag is represented
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)


tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)


model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")


print("Trained")
