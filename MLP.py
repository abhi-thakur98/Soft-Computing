import pandas as pd
import numpy as np
import random
from math import exp
from matplotlib import pyplot as plt

# Logistic Function
def logistic(x):
    return 1/(1+exp(-x))

def fun(a):
    a = list(a)
    ind = a.index(max(a))
    return [0 for i in range(ind)]+[1]+[0 for i in range(len(a)-ind-1)]

# Reading the data using Pandas
df = pd.read_csv('data.csv')
inputs = df[df.columns[1:-1]].values
labels = df[df.columns[-1]].to_numpy()
weights1 = np.asarray([[random.random() for i in range(len(inputs[0])+1)] for j in range(len(set(labels)) + 3)])
weights2 = np.asarray([[random.random() for i in range(len(set(labels))+3)] for j in range(len(set(labels)))])
output = np.asarray([[0]*(i-1)+[1]+[0]*(len(set(labels)) - i) for i in labels])

# Spliting the data into training and testing set
random.seed(60)
train = sorted(random.sample(range(0,len(inputs)),int(0.7*len(inputs))))
test = [i for i in range(len(inputs)) if i not in train]
train_input = np.insert(inputs[train], 0, 1, axis=1)
test_input = np.insert(inputs[test], 0, 1, axis=1)
train_output = output[train]
test_output = output[test]
delta = np.asarray([0.0]*len(set(labels)))

# Learning Process
x_axis = []
accuracy = []
for epoch in range(101):
    count = 0
    for i in train_input:
        old_weights = weights2.copy()
        output1 = np.asarray([logistic(j) for j in i.dot(np.transpose(weights1))])
        output = np.asarray([logistic(j) for j in output1.dot(np.transpose(weights2))])

        diff = train_output[count] - output
        for j in range(len(diff)):
            delta[j] = diff[j] * output[j] * (1-output[j])
            weights2[j] += delta[j] * output1
                
        diff = 1 - output1
        for j in range(len(diff)):
            delta2 = diff[j] * output1[j] * delta.dot(old_weights)[j]
            weights1[j] += delta2 * i
        count += 1
    p_output1 = np.asarray([[logistic(j) for j in i] for i in test_input.dot(np.transpose(weights1))])
    p_output2 = np.asarray([[logistic(j) for j in i] for i in p_output1.dot(np.transpose(weights2))])
    x_axis.append(epoch+1)
    predicted_output = np.asarray([fun(i) for i in p_output2])
    accuracy.append((sum([1 for i in test_output == predicted_output if False not in i])/len(test_output))*100)
print('No. of epochs -->',epoch+1)

# Plotting the graph between accuracy and epochs
fig,ax= plt.subplots(figsize=(15,5))
plt.plot(x_axis,accuracy)
ax.set_xticks(x_axis[::5])
plt.xlabel('Epochs')
plt.ylabel('Accuracy %')
ax.set_yticks(range(0,101,10))
plt.grid()
plt.show()

# Final Accuracy
print('Accuracy =', round((sum([1 for i in test_output == predicted_output if False not in i])/len(test_output))*100,2),'%')




