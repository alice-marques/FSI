from mnist import MNIST
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
mndata = MNIST('/home/andre/UnB/FSI/K-NN')
print('Loading trainig data...')
imagesa,labels = mndata.load_training()
print('Loading testing data...')
imagesTesta, labelsTest = mndata.load_testing()
images = np.reshape(imagesa,(60000,28,28))
features = np.zeros((60000,28))
i = 0
j = 0
print('Making the sums of each row from the trainig file...')
for num in images:
    for row in num:
        for elements in row:
            features[i][j]+= elements
        j += 1
    i += 1
    j = 0

i = 0
imagesTest = np.reshape(imagesTesta,(10000,28,28))
featuresTest = np.zeros((10000,28))
print('Making the sums of each row from the testing file...')
for num in imagesTest:
    for row in num:
        for elements in row:
            featuresTest[i][j]+= elements
        j += 1
    i += 1
    j = 0
print('Making models and aplying data...')
modelSeparate = KNeighborsClassifier(n_neighbors=245)
modelSeparate.fit(imagesa,labels)
modelSum = KNeighborsClassifier(n_neighbors=245) #trecho obtido no site https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn
modelSum.fit(features,labels)
print('Making predictions based on 245 Nearest Neighbors...')
predictionSeparate = modelSeparate.predict(imagesTesta)
predictionSum = modelSum.predict(featuresTest)
correctAnswers = 0
print('Calculating succes percentage for untreated data...')
for num in predictionSeparate:
    if num == labelsTest[j]:
        correctAnswers += 1
    j+=1
print('Without making sums we had a',correctAnswers/100,'% succesrating.')
correctAnswers = 0
print('Calculating succes percentage for the summed rows...')
j = 0
for numa in predictionSum:
    if numa == labelsTest[j]:
        correctAnswers += 1
    j+=1
print('Making the sums we had a ',correctAnswers/100,'% success rating.')
