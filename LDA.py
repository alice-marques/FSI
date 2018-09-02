from mnist import MNIST
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
mndata = MNIST('/home/andre/UnB/FSI/Trabalho1')
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
modelSimple = LinearDiscriminantAnalysis()
#Seguindo a documentacao encontrada em: https://goo.gl/q5AGAr
modelSimple.fit(imagesa,labels)
modelSum = LinearDiscriminantAnalysis()
modelSum.fit(features, labels)
print('Calculating grouping...')
resultsSimple = modelSimple.predict(imagesTesta)
resultsSum = modelSum.predict(featuresTest)
correctAnswers = 0
print('Calculating success percentage for untreated model...')
for num in resultsSimple:
    if num == labelsTest[j]:
        correctAnswers += 1
    j+=1
print('For the simple model we had a',correctAnswers/100,'% success rating')
print('Calculating success percentage for treated model...')
correctAnswers = 0
j = 0
for num in resultsSum:
    if num == labelsTest[j]:
        correctAnswers += 1
    j+=1
print('For the treated model we had a',correctAnswers/100,'% success rating')
