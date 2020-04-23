
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tensorflow
import tflearn
import random
import json
import pickle

with open('neuroBotTrain.json') as file:
  data = json.load(file)

#if data.picle exists then don't train and just load previous data otherwise train the bot with the json file
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    #loops through the intents in the json file
    for intent in data['intents']:
        #loops through patterns in the json file
        for pattern in intent['patterns']:
            #gets list of all of the root words in the patterns
            wrds = nltk.word_tokenize(pattern)

            #puts the tokenized words from the patterns and puts them in words list
            words.extend(wrds)

            #adds patterns to docs_x list
            docs_x.append(wrds)

            #adds tags of the associated pattern to docs_y
            docs_y.append(intent["tag"])
        #if the tag is not already in labels then it gets added
        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    #word stemming so that the "happening" will be the same as "happen" (Puts the words in lower case and gets rid of '?')
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]

    #removes duplicates and sorts list
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
        #stems the words in doc_x
        wrds = [stemmer.stem(w.lower()) for w in doc]

        #If a word is in the main word list then it is represented as a 1 otherwise 0
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        #looks through labels list get location of the tag and set that value to 1 in output row
        output_row[labels.index(docs_y[x])] = 1

        #adds the enumerated lists to training and output
        training.append(bag)
        output.append(output_row)

    #needs to by numpy array to use tflearn
    training = numpy.array(training)
    output = numpy.array(output)

    #make the pickle file
    with open("data.pickle","wb")as f:
        pickle.dump((words, labels, training, output), f)




tensorflow.reset_default_graph()

#defines input shape; input layer length of input data
net = tflearn.input_data(shape=[None, len(training[0])])
#add fully connected layer to neural network (layer is made up of 8 neurons)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)

#allows us to get probabilities for each neuron in the layer; output layer; gets what the response should be based on the probabilities of the tags
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

#trains the model
model = tflearn.DNN(net)

#if model.tflearn already exists does not need to fit the model
try:
    model.load("model.tflearn")
except:
    #pass the model the training data and saves the model
    model.fit(training, output, n_epoch=1500, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(s, words):
    #create blank bag of words list
    bag = [0 for _ in range(len(words))]

    #gets list of tokenized words
    s_words = nltk.word_tokenize(s)
    #stems the words
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    #loops through the tokenized and stemmed words
    for se in s_words:
        for i, w in enumerate(words):
            #if current words is equal to the word in our sentence then index gets set as 1
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
