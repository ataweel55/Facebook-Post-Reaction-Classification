import pandas as pd
import numpy as np
import h5py
from keras.models import load_model
from keras.models import Model, Sequential
from keras.layers import Activation, Conv1D, MaxPooling1D, Flatten, Dense, Input, concatenate
from keras.callbacks import ModelCheckpoint
from nltk.tokenize import word_tokenize



# hyperparameters
param_num_epochs = 100
param_batch_size = 50
param_post_size = 100
param_embedding_size = 100

# input dimensions for first convolutional Layer - WITHOUT BATCH!
input_dim = (param_post_size, param_embedding_size)

num_classes = 5
data_path = "/Users/mehmetnur/Desktop/ML Python/webmining/data/"


def readCsv(file_name):
    path = data_path + file_name
    train = pd.read_csv(path, encoding="ISO-8859-1", usecols=['post', 'ANGRY', 'LOVE', 'HAHA', 'WOW', 'SAD'])
    return train


def printRecord(n):
    print(trainData['post'][n] + " | angrys: " + str(trainData['ANGRY'][n]) + " | loves: " + str(trainData['LOVE'][n]) + " | hahas: " + str(trainData['HAHA'][n]) + " | wows: " + str(trainData['WOW'][n]) + " | sads: " + str(trainData['SAD'][n]))

    
def generate_vocabulary(data):

    # generate dictionary
    word_id = 0
    word_to_id = {}  
    
    print("Start generating vocabulary")
    for d in data:

       
        for post in d['post']:

            # tokenize text
            tokens = word_tokenize(str(post))

            # add word to dictionary
            for token in tokens:
                if(len(token) > 2):
                    if token not in word_to_id:
                        word_to_id[token] = word_id
                        word_id = word_id + 1

    return word_to_id


def read_embeddings(filename):
    path = data_path + filename
    embedding_dict = {}
    with open(path) as f:
        content = f.readlines()
        for line in content:
            tokens = line.split(' ')
            key = tokens[0]
            value = tokens[1:]
            embedding_dict[key] = value
    return embedding_dict


def postDict_to_shape(trainDataName, embedding, count_words):

    trainData = readCsv(trainDataName)
    count_posts = len(trainData['post'])

    posts = np.zeros((count_posts, count_words, param_embedding_size))
    reactions = np.zeros((count_posts, num_classes))

    for postIndex in range(count_posts):
        
        # tokenize post
        tokens = word_tokenize(str(trainData['post'][postIndex]))

        # prepare post embedding list
        wordCounter = 0
        for tokenIndex in range(len(tokens)):
            if (tokens[tokenIndex].lower() in embedding.keys()) and (len(tokens[tokenIndex]) > 2):
                posts[postIndex][wordCounter] = embedding[tokens[tokenIndex].lower()]
                wordCounter += 1

            if wordCounter == count_words:
                break

        # prepare reactions list
        reactions[postIndex][0] = trainData['ANGRY'][postIndex]
        reactions[postIndex][1] = trainData['SAD'][postIndex]
        reactions[postIndex][2] = trainData['WOW'][postIndex]         
        reactions[postIndex][3] = trainData['HAHA'][postIndex]
        reactions[postIndex][4] = trainData['LOVE'][postIndex]
    
    return posts, reactions


class BatchGenerator(object):

    # input_data dimensions: [num_posts][num_words][embeddings]
    # output_data dimensions: [num_posts][reactions]
    
    def __init__(self, input_data, reaction_data, batch_size, post_length, embedding_size, num_classes):
        self.input_data = input_data
        self.reaction_data = reaction_data
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.post_length = post_length
        
        self.current_id = 0


    # x [number of phrases (batch size), number of words for each phrase given for prediction]
    # y additionally has the number of dictionary dimensions (used for softmax later) 
    # --> for each target word, which is going to be predicted, ground truth is needed (one-hot vector)

    def generate(self):
        x = np.zeros((self.batch_size, self.post_length, self.embedding_size))
        y = np.zeros((self.batch_size, self.num_classes))
        while True:
            for i in range(self.batch_size):

                # prepare input and output data  
                x[i, :, :] = self.input_data[self.current_id]   
                y[i, :] = self.reaction_data[self.current_id]
                
                # increase current id
                self.current_id += 1
                if self.current_id + 1 >= len(self.input_data):
                    # reset the index back to the start of the data set
                    self.current_id = 0
            yield x, y


def prepareData():
    # Data preparation
    print("Prepare data...")

    # generate embedding dictionary
    embeddings = read_embeddings("glove.twitter.27B.100d.txt")
    print("embeddings dictionary generated!")

    # convert raw post data to final shape - [posts][words][embedding]
    x_train, y_train = postDict_to_shape("BalancedTrain.csv", embeddings, count_words = 100)
    print("Training data prepared!")

    x_valid, y_valid = postDict_to_shape("validation.csv", embeddings, count_words = 100)
    print("Validation data prepared!")   
    
    return x_train, y_train, x_valid, y_valid 

def generate_CNN():
    model = Sequential()
    model.add(Conv1D(100, kernel_size=(3), activation='relu', input_shape=input_dim))
    model.add(MaxPooling1D(pool_size=(3)))
    model.add(Flatten())
    model.add(Dense(300, activation='sigmoid'))
    model.add(Dense(num_classes, activation='sigmoid'))
    return model


def generate_CNN2():
    
    inp = Input(input_dim)

    conv_branch_1 = Conv1D(300, kernel_size=(2), activation='linear')(inp)
    conv_branch_1 = MaxPooling1D(pool_size=(2))(conv_branch_1)

    conv_branch_2 = Conv1D(300, kernel_size=(3), activation='linear')(inp)
    conv_branch_2 = MaxPooling1D(pool_size=(2))(conv_branch_2)

    conv_branch_3 = Conv1D(300, kernel_size=(5), activation='linear')(inp)
    conv_branch_3 = MaxPooling1D(pool_size=(2))(conv_branch_3)

    merged = concatenate([conv_branch_1, conv_branch_2, conv_branch_3], axis=1)
    merged = Flatten()(merged)

    out = Dense(100, activation='tanh')(merged)
    out = Dense(num_classes, activation='tanh')(out)

    model = Model(inp, out)

    return model

def train(model):
    # prepare and filter Data
    x_train, y_train, x_valid, y_valid = prepareData()

    model.summary()
    
    # load weights
    # model = load("model-50.hdf5")
    
    # generate training data generator
    train_data_generator = BatchGenerator(x_train, y_train, batch_size = param_batch_size, post_length = param_post_size, embedding_size = param_embedding_size, num_classes=5)
    valid_data_generator = BatchGenerator(x_valid, y_valid, batch_size = param_batch_size, post_length = param_post_size, embedding_size = param_embedding_size, num_classes=5)

    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mae', 'binary_accuracy'])
    
    # checkpoint after each epoch
    checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)

    # train model  
    model.fit_generator(train_data_generator.generate(), len(x_train)//(param_batch_size), param_num_epochs,
                            validation_data=valid_data_generator.generate(),
                            validation_steps=len(x_valid)//(param_batch_size), callbacks=[checkpointer])


def load(weight_filename):
    weight_path = data_path + weight_filename

    model = generate_CNN()

    model.load_weights(weight_path)

    return model


# generate model
model = generate_CNN()
# model = generate_CNN2()

# load weights
# model = load(filename)
train(model)
