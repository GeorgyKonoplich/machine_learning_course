import random
import math
import operator
import numpy

def loadData():
	with open('input.txt') as f:
		array = []
		for line in f:
			float_list = [float(i) for i in line.split()]
			array.append(float_list)
	return array

def splitData(array, split, trainingSet=[] , testSet=[]):
	for x in range(len(array)):
		if (random.random() < split):
			trainingSet.append(array[x])
		else:
			testSet.append(array[x])

def euclideanDistance(point1, point2):
	distance = pow((point1[0] - point2[0]), 2) + pow((point1[1] - point2[1]), 2)
	return math.sqrt(distance)

# distances = {point, dist}
def applyKernelFunction(distances):
	array = []
	for x in range(len(distances) - 1):
		d = distances[x][1] / distances[-1][1]
		#print (distances[x][1],' ', distances[-1][1], ' ', distances[x][0][2])		
		#array.append([distances[x][0], 1 - d])   #triangular
		array.append([distances[x][0], (1 - pow(d,2))*3/4])  #epanechnikov
	#print ('end')
	array.sort(key=operator.itemgetter(1))
	return array

# neighbors - {point, kernel dist}
def getNeighbors(point, train80, k):
	#print(point)
	distances = []
	for x in range(len(train80)):
		dist = euclideanDistance(point, train80[x])
		distances.append((train80[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k + 1):
		neighbors.append(distances[x])
#	if (k == 1):
#		print neighbors
	
	return applyKernelFunction(neighbors)
# k - classificator
# test20 - test list
# train80 - train list
def classifyKNN(k, test20, train80):
	count = 0
	for x in range(len(test20)):
		class1, class2 = 0, 0
		neighbors = getNeighbors(test20[x], train80, k)
			#	print neighbors
		for y in range(len(neighbors)):
			if (neighbors[y][0][2] == 0.0):
				class1 += neighbors[y][1]
			else:
				class2 += neighbors[y][1]
	#	if (class1 == class2):
	#srint (class1, ',', class2, ',', test20[x][2], ',', k)
		if class1 > class2 and test20[x][2] == 0.0:
			count += 1
		else:
			if (class2 > class1 and test20[x][2] == 1.0):
				count += 1
#		print (count)
	#print (count, ' ', len(test20))
			
	return count


def splitTrainData(data, k):
	count = 0
	for i in range(1, 6):
		test20 = []
		test20ind = []
		train80 = []
		train80ind = []
		length = len(data)
		chunks = length / 5

		for j in range(length):
			if (j <= (i * chunks)) and (j >= ((i - 1) * chunks)):
				test20.append(data[j])
				test20ind.append(j)
			else:
				train80.append(data[j])
				train80ind.append(j)

	#	print ('chunks', ',', 'train20ind = ', test20ind, ',', 'train80ind = ', train80ind)
		count += classifyKNN(k, test20, train80)
	return float(count) / float(len(data))

def main():
	array = loadData()
	random.shuffle(array)
	trainingSet=[]
	testSet=[]
	split = 0.8
	splitData(array, split, trainingSet, testSet)

	length = (len(trainingSet) * 4) / 5
	answer = []
	for k in range(1, 10) :
		accuracy = splitTrainData(trainingSet, k)
		answer.append([accuracy, k])

	#print answer
	answer.sort(key=operator.itemgetter(0))
	print ('PRELIMINARY ANSWER', answer[-1])
	total = classifyKNN(answer[-1][1], testSet, trainingSet)
	#total = classifyKNN(9, testSet, trainingSet)
	print ('FINAL ANSWER , k = ', answer[-1][1], ', loss percent = ', 1 - float(total)/float(len(testSet)))
	

main()