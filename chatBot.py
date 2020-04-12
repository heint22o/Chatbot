
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tensorflow
import tflearn
import random
import json
import pickle

with open('intents.json') as file:
  data = json.load(file)

# will only retrain model and create data if it hasn't already been done
try:
    with open("data.pickle", "rb") as f:
         words, label, training, output = pickle.load(f)
except:
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

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")
print("Trained")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)

def chat():
    print("start talking with the bot (type quit to stop)")
    while True:
        # gets input from user
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        # convert input into numbers (bag of words) and get a prediction
        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        # find most probable class and pick a responses from it
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))
chat()
