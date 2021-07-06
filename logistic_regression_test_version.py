import json
import numpy as np
import re
import random
import time

# starting time
start = time.time()   
#Obtaining dataset
dataset = np.empty((0,2))
# Opening JSON file
f = open('Sarcasm_Headlines_Dataset - short.json',)
  
# returns JSON object as 
# a dictionary
data = json.load(f)
  
# Iterating through the json
# list
for i in data['dataset_sarcasm']:
    i = str(i)
    i = i.replace("{","")
    i = i.replace("}","")
    i = i.replace("'is_sarcastic': ","")
    i = i.replace(" 'headline': ","")
    data = i.split(",")
    data = data[:-1]
    val_1 = data[0]
    val_2 = data[1]
    dataset = np.append(dataset, np.array([[val_1,val_2]]), axis=0)
     
# Closing file
f.close()


#bag of words model

#Get array of sentences
def get_headlines(dataset):
    headlines = np.array([])
    targets = np.array([])
    for j in dataset:
        headlines = np.append(headlines,j[1])
        targets = np.append(targets,int(j[0]))
    
    return headlines,targets

headlines,targets = get_headlines(dataset)

#Tokenize a sentence:
def word_extraction(sentence):
    stop_words = ["a","the","is"]
    words = re.sub("[^\w]", " ",  sentence).split()
    new_text = [w.lower() for w in words if w not in stop_words]
    
    return new_text

def tokenize(headline):
    words = []
    for s in headline:
        w = word_extraction(s)
        words.extend(w)
        words = sorted(list(set(words)))
    return words    


def generate_bow(headlines):
    j = 0
    vocab = tokenize(headlines)
    arr_bag_vector = np.zeros((len(headlines),len(vocab)))
    for headline in headlines:
        words = word_extraction(headline)
        bag_vector = np.zeros((len(vocab),1))
        for letter in words:
            for i,word in enumerate(vocab):
                if word == letter:
                    bag_vector[i] += 1
        
        
        for k in range(len(vocab)):
            arr_bag_vector[j][k] = bag_vector[k]
                
        j = j + 1
    return arr_bag_vector

#logistic Regression

#datapoint
x = generate_bow(headlines)
x = np.insert(x,0,1,axis=1)

#different datasets - 80% temp data, 20% test data - 80% of temp = training & 20% of temp = validation
temp_data = x[:768]
training_data = temp_data[:600]
validation_data = temp_data[600:768]
test_data = x[768:960]

def initialize_weights(n):
    size = int(n.size / len(n))
    weights = np.ones((size,1))

    for k in range(size):
        weights[k] = random.uniform(-0.5, 0.5)
    return weights

#datasets weights
training_weights = initialize_weights(training_data)
validation_weights = initialize_weights(validation_data)
test_weights = initialize_weights(test_data)

#classes of datasets
y = targets
temp_labels = y[:768]
training_labels = temp_labels[:600]
#print(training_labels.shape)
validation_labels = temp_labels[600:768]
test_labels = y[768:960]

#prediction
def prediction(weights,x):
    z = np.dot(np.transpose(weights),x)
    return sigmoid(z)
#sigmpid function for prediction
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
#gradient descent
def update_weights(weights,x,y):
    lr = 0.1
    reg_lambda = 0.02
    for k in range(20):
        for j in range(len(x)):
            predict = prediction(weights,x[j])

            for i in range(len(weights)):
                w_temp = weights[i] - lr * ((predict - y[j]) * x[j][i] + (reg_lambda * weights[i]))
                weights[i] = w_temp
    return weights

#prediction using new weights
def new_prediction(new_weights,x):
    predict_class = np.array([])
    for k in range(len(x)):
         z = np.dot(np.transpose(new_weights),x[k])
         #print("z", sigmoid(z))
         
         if(sigmoid(z) < 0.5):
             #print("class: ",0)
             predict_class = np.append(predict_class,0)
         else:
             #print("class: ",1)
             predict_class = np.append(predict_class,1)
    
    return predict_class

#accuracy for new predictons
def accuracy(class_predictions,y):
    #number of correct predictions
    hit = 0
    #number of wrong predictions
    miss = 0
    #number of sarcastic label that were sarcastic
    sarcastic = 0
    #number of sarcastic label that were serious
    serious = 0
    #number of serious label that were serious
    not_sarcastic = 0
    #number of serious label that were sarcastic
    not_serious = 0
    
    for i in range(len(y)):
        if(class_predictions[i] == y[i]):
            hit = hit + 1
            if(class_predictions[i] == 1):
                sarcastic = sarcastic + 1
            else:
                serious = serious + 1
        else:
            miss = miss + 1
            if(class_predictions[i] == 1):
                not_sarcastic = not_sarcastic + 1
            else:
                not_serious = not_serious + 1
            
    print("Hit: ",hit)
    print("Miss: ",miss)
    print("sarcastic: ", sarcastic)
    print("not sarcastic: ", not_sarcastic)
    print("serious: ", serious)
    print("not serious: ", not_serious)
    
    return (hit / (hit + miss)) * 100


#training dataset
print("training data")
training_prediction = prediction(training_weights,np.transpose(training_data))
training_new_weights = update_weights(training_weights,training_data,training_labels)
training_new_prediction = new_prediction(training_new_weights,training_data)
training_accuracy = accuracy(training_new_prediction,training_labels)
print(training_accuracy)

#validation dataset
print("validation data")
validation_prediction = prediction(validation_weights,np.transpose(validation_data))
validation_new_weights = update_weights(validation_weights,validation_data,validation_labels)
validation_new_prediction = new_prediction(training_new_weights,validation_data)
validation_accuracy = accuracy(validation_new_prediction,validation_labels)
print(validation_accuracy)

#test dataset
print("test data")
test_prediction = prediction(test_weights,np.transpose(test_data))
#test_new_weights = update_weights(test_weights,test_data,test_labels)
test_new_prediction = new_prediction(validation_new_weights,test_data)
test_accuracy = accuracy(test_new_prediction,test_labels)
print(test_accuracy)

#randomly guess predictions
print("guess: ")
arr_guess = np.random.randint(2,size=len(test_new_prediction))
print(accuracy(arr_guess,test_labels))

# end time
end = time.time()

# total time taken
print(f"Runtime of the program is {end - start}")