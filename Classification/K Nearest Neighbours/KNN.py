import numpy as np
import operator

'''
Attribute Information:

1. Sample code number: id number 
2. Clump Thickness: 1 - 10 
3. Uniformity of Cell Size: 1 - 10 
4. Uniformity of Cell Shape: 1 - 10 
5. Marginal Adhesion: 1 - 10 
6. Single Epithelial Cell Size: 1 - 10  
7. Bare Nuclei: 1 - 10 
8. Bland Chromatin: 1 - 10 
9. Normal Nucleoli: 1 - 10 
10. Mitoses: 1 - 10 
11. Class 

Dataset taken from the UCI machine learning repository.
'''


def getData(fileName):
    labels = []
    data = np.genfromtxt(fileName,delimiter = ',')
    for row in data:
        labels.append(row[-1])
    data = np.delete(data,10,1)   
    data = np.delete(data,0,1)
    return data,labels

def classify(inX,dataSet,labels,k):

    #We must first calculate the distance between each point in the dataset and the given point for which we must determine the label
    #We achieve this by calculating the Euclidean distance for the aforementioned points

       
    dataSetSize = dataSet.shape[0]                                                  # arr.shape[i] returns the i'th dimension of the numpy array ex- 3*4, then shape[0] = 3
    
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet                               # tile repeats the first parameter array by the second parameter times. So here, the inX(value we need to classify) is repeated (dataSetSize,1) times and the dataset is then subtracted from it to give the difference matrix
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()                    
    classCount = {}

    #Now we must have a vote between the k nearest neighbour points to the given point. The same is implemented below 

    for i in range(k):                                                      
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return 'Benign' if sortedClassCount[0][0]==2 else 'Malignant'

if __name__ == "__main__":
    fileName = "breast-cancer-wisconsin.csv"
    dataSet,labels = getData(fileName)
    inX = [5,1,3,4,6,5,2,3,5]
    classX = classify(inX,dataSet,labels,25)
    print(classX)