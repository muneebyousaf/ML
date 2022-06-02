from numpy import argmax
from math import exp
# define data
data = [1, 3, 2]
# calculate the max of the list
result = max(data)
print(" Max value in the  data: %d" %result)


# example of the argmax of a list of numbers

# define data
data = [1, 3, 2]
# calculate the argmax of the list
result = argmax(data)
print("The index of max value in the data: %d" %result)




# calculate the softmax of a vector
def softmax(vector):
	e = exp(vector)
	return e / e.sum()

# define data
data = [1, 3, 2]
# convert list of numbers to a list of probabilities
result = softmax(data)
# report the probabilities
print(result)
# report the sum of the probabilities
print(sum(result))