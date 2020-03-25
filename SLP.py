import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt

# Reading the data using Pandas
df = pd.read_csv('data.csv')
inputs = df[df.columns[1:-1]].values
labels = df[df.columns[-1]].to_numpy()
weights = np.asarray([[0 for i in range(len(inputs[0])+1)] for j in range(len(set(labels)))])
output = np.asarray([[0]*(i-1)+[1]+[0]*(len(set(labels)) - i) for i in labels])

# Spliting the data into training and testing set
random.seed(60)
train = sorted(random.sample(range(0,len(inputs)),int(0.7*len(inputs))))
test = [i for i in range(len(inputs)) if i not in train]
train_input = np.insert(inputs[train], 0, 1, axis=1)
test_input = np.insert(inputs[test], 0, 1, axis=1)
train_output = output[train]
test_output = output[test]

# Learning Process
x_axis = []
errors = []
accuracy = []
for epoch in range(100):
    count = 0
    e = 0
    for i in train_input:
        output = np.asarray([int(j>0) for j in i.dot(np.transpose(weights))])
        diff = train_output[count] - output
        for j in range(len(diff)):
            if diff[j] != 0:
                e += 1
                weights[j] += diff[j]*i
        count += 1
    x_axis.append(epoch+1)
    errors.append(e)
    predicted_output = np.asarray([[int(j>0) for j in i] for i in test_input.dot(np.transpose(weights))])
    accuracy.append((sum([1 for i in test_output == predicted_output if False not in i])/len(test_output))*100)
    if e == 0:
        break
print('No. of epochs -->',epoch+1)

# Plotting the graph between errors and epochs
fig,ax= plt.subplots(figsize=(15,5))
plt.plot(x_axis,errors)
ax.set_xticks(x_axis[::2])
plt.xlabel('Epochs')
plt.ylabel('No. of errors')
plt.grid()
plt.show()

# Plotting the graph between accuracy and epochs
fig,ax= plt.subplots(figsize=(15,5))
plt.plot(x_axis,accuracy)
ax.set_xticks(x_axis[::2])
plt.xlabel('Epochs')
plt.ylabel('Accuracy %')
ax.set_yticks(range(0,101,10))
plt.grid()
plt.show()

# Final Accuracy
print('Accuracy =',round((1 - sum([1 for i in test_output == predicted_output if False in i])/len(test_input))*100,2),'%')



