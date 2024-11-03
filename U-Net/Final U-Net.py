#Declaring the path to images
image_dir = 'C:/Users/rohit/Desktop/PythonSelfLearn/DeepLearningProject/U-Net/malignant'

#SEtting up two new folders
import os 
origImagePath = 'C:/Users/rohit/Desktop/PythonSelfLearn/DeepLearningProject/U-Net/malignant/origImage'
maskImagePath = 'C:/Users/rohit/Desktop/PythonSelfLearn/DeepLearningProject/U-Net/malignant/maskImage'
os.makedirs(origImagePath, exist_ok = True)
os.makedirs(maskImagePath, exist_ok = True)


#Separating image and its mask
from PIL import Image

reSize = (256,256)
for file in os.listdir(image_dir):
    if file.endswith('.png'):
        if '_mask' in file:
            maskPath = image_dir + '/' + file
            mask = Image.open(maskPath).convert('L')
            mask = mask.resize((256,256))
            file = file.replace('_mask.','.')
            newPath = maskImagePath + '/' + file
            
            mask.save(newPath)
            print(f"Processed Mask: {file}")
        else:
            origPath = image_dir + '/' + file
            origImage = Image.open(origPath).convert('L')
            origImage = origImage.resize((256,256))
            newOrigImagePath = origImagePath + '/' + file
            origImage.save(newOrigImagePath)
            print(f"Processed Original: {file}")
			
#Printing the number of images and mask to ensure its equal
def FindNumberOfPhotos(path):
    count = 0
    for item in os.listdir(path):
        count += 1
    return count

noOfOrigImage = FindNumberOfPhotos(origImagePath)
print(noOfOrigImage)
noOfMaskImage = FindNumberOfPhotos(maskImagePath)
print(noOfMaskImage)    

import torchvision.transforms as transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

from sklearn.model_selection import train_test_split
imageFiles = sorted(os.listdir(origImagePath))
maskFiles = sorted(os.listdir(maskImagePath))


#Train and Test split
if len(imageFiles) == len(maskFiles):
    print("Ready to be divided into Test and Train data")

trainImages, testImages, trainMasks, testMasks = train_test_split(imageFiles, maskFiles, test_size = 0.2, random_state = 42)

os.makedirs('C:/Users/rohit/Desktop/PythonSelfLearn/DeepLearningProject/U-Net/malignant/imageTest',exist_ok = True)
os.makedirs('C:/Users/rohit/Desktop/PythonSelfLearn/DeepLearningProject/U-Net/malignant/maskTest',exist_ok = True)
os.makedirs('C:/Users/rohit/Desktop/PythonSelfLearn/DeepLearningProject/U-Net/malignant/imageTrain',exist_ok = True)
os.makedirs('C:/Users/rohit/Desktop/PythonSelfLearn/DeepLearningProject/U-Net/malignant/maskTrain',exist_ok = True)


#Copying all images and masks to separate folders
for img in testImages:
    testImagePath = 'C:/Users/rohit/Desktop/PythonSelfLearn/DeepLearningProject/U-Net/malignant/origImage' + '/' + img
    testImage = Image.open(testImagePath)
    newTestImagesPath = 'C:/Users/rohit/Desktop/PythonSelfLearn/DeepLearningProject/U-Net/malignant/imageTest' + '/' + img
    testImage.save(newTestImagesPath)
    
for img in testMasks:
    testMaskPath = 'C:/Users/rohit/Desktop/PythonSelfLearn/DeepLearningProject/U-Net/malignant/maskImage' + '/' + img
    testMask = Image.open(testMaskPath)
    newTestMaskPath = 'C:/Users/rohit/Desktop/PythonSelfLearn/DeepLearningProject/U-Net/malignant/maskTest' + '/' + img
    testMask.save(newTestMaskPath)

for img in trainImages:
    trainImagePath = 'C:/Users/rohit/Desktop/PythonSelfLearn/DeepLearningProject/U-Net/malignant/origImage' + '/' + img
    trainImage = Image.open(trainImagePath)
    newTrainImagesPath = 'C:/Users/rohit/Desktop/PythonSelfLearn/DeepLearningProject/U-Net/malignant/imageTrain' + '/' + img
    trainImage.save(newTrainImagesPath)
    
for img in trainMasks:
    trainMaskPath = 'C:/Users/rohit/Desktop/PythonSelfLearn/DeepLearningProject/U-Net/malignant/maskImage' + '/' + img
    trainMask = Image.open(trainMaskPath)
    newTrainMaskPath = 'C:/Users/rohit/Desktop/PythonSelfLearn/DeepLearningProject/U-Net/malignant/maskTrain' + '/' + img
    trainMask.save(newTrainMaskPath)
    
 #Printing the number of Train and Test data
noOfTrainImage = FindNumberOfPhotos('C:/Users/rohit/Desktop/PythonSelfLearn/DeepLearningProject/U-Net/malignant/imageTrain')
print(noOfTrainImage)
noOfTrainMask = FindNumberOfPhotos('C:/Users/rohit/Desktop/PythonSelfLearn/DeepLearningProject/U-Net/malignant/maskTrain')
print(noOfTrainMask)   
noOfTestImage = FindNumberOfPhotos('C:/Users/rohit/Desktop/PythonSelfLearn/DeepLearningProject/U-Net/malignant/imageTest')
print(noOfTestImage)
noOfTestMask = FindNumberOfPhotos('C:/Users/rohit/Desktop/PythonSelfLearn/DeepLearningProject/U-Net/malignant/maskTest')
print(noOfTestMask) 



#Load the images in DataLoader
from torch.utils.data import DataLoader
imageNames = [f for f in os.listdir('C:/Users/rohit/Desktop/PythonSelfLearn/DeepLearningProject/U-Net/malignant/imageTrain')]
maskNames = [f for f in imageNames]
trainCombinedDataset = []
for imageName, maskName in zip(imageNames,maskNames):
    imagePath = 'C:/Users/rohit/Desktop/PythonSelfLearn/DeepLearningProject/U-Net/malignant/imageTrain' + '/' + imageName
    maskPath = 'C:/Users/rohit/Desktop/PythonSelfLearn/DeepLearningProject/U-Net/malignant/maskTrain' + '/' + maskName
    image = Image.open(imagePath)
    mask = Image.open(maskPath)
    image = transform(image)
    mask = transform(mask)
    trainCombinedDataset.append((image,mask))

imageNames = [f for f in os.listdir('C:/Users/rohit/Desktop/PythonSelfLearn/DeepLearningProject/U-Net/malignant/imageTest')]
maskNames = [f for f in imageNames]
testCombinedDataset = []
for imageName, maskName in zip(imageNames,maskNames):
    imagePath = 'C:/Users/rohit/Desktop/PythonSelfLearn/DeepLearningProject/U-Net/malignant/imageTest' + '/' + imageName
    maskPath = 'C:/Users/rohit/Desktop/PythonSelfLearn/DeepLearningProject/U-Net/malignant/maskTest' + '/' + maskName
    image = Image.open(imagePath)
    mask = Image.open(maskPath)
    image = transform(image)
    mask = transform(mask)
    testCombinedDataset.append((image,mask))

trainDataloader = DataLoader(trainCombinedDataset, batch_size = 12, shuffle = True)
testDataLoader = DataLoader(testCombinedDataset,batch_size = 12, shuffle = True)


#Printing the contents of Each Batch
for batchNum , (images, masks) in enumerate(trainDataloader):
    print(f"TrainBatch {batchNum + 1}")
    print(f"  Images: {images.shape}") 
    print(f"  Masks: {masks.shape}")

for batchNum , (images, masks) in enumerate(testDataLoader):
    print(f"TestBatch {batchNum + 1}")
    print(f"  Images: {images.shape}") 
    print(f"  Masks: {masks.shape}")
    
    
 def unet_model():
    inputs  = Input((256,256,1))
     
    s1 = Conv2D(64,3,activation = 'relu' , padding = 'same')(inputs)
    s1 = Conv2D(64,3,activation = 'relu' , padding = 'same')(s1)
    p1 = MaxPooling2D(pool_size = (2,2))(s1)

    s2 = Conv2D(128,3,activation = 'relu' , padding = 'same')(p1)
    s2 = Conv2D(128,3,activation = 'relu' , padding = 'same')(s2)
    p2 = MaxPooling2D(pool_size = (2,2))(s2)

    s3 = Conv2D(256,3,activation = 'relu' , padding = 'same')(p2)
    s3 = Conv2D(256,3,activation = 'relu' , padding = 'same')(s3)
    p3 = MaxPooling2D(pool_size = (2,2))(s3)

    s4 = Conv2D(512,3,activation = 'relu' , padding = 'same')(p3)
    s4 = Conv2D(512,3,activation = 'relu' , padding = 'same')(s4)
    p4 = MaxPooling2D(pool_size = (2,2))(s4)

    b1 = Conv2D(1024,3,activation = 'relu' , padding = 'same')(p4)
    b1 = Conv2D(1024,3,activation = 'relu' , padding = 'same')(b1)


    d1 = Conv2D(512,2,activation = 'relu' , padding = 'same')(UpSampling2D(size = (2,2))(b1))
    d1 = concatenate([s4,d1], axis = 3)
    d1 = Conv2D(512,3,activation = 'relu' , padding = 'same')(d1)
    d1 = Conv2D(512,3,activation = 'relu' , padding = 'same')(d1)

    d2 = Conv2D(256,2,activation = 'relu' , padding = 'same')(UpSampling2D(size = (2,2))(d1))
    d2 = concatenate([s3,d2], axis = 3)
    d2 = Conv2D(256,3,activation = 'relu' , padding = 'same')(d2)
    d2 = Conv2D(256,3,activation = 'relu' , padding = 'same')(d2)

    d3 = Conv2D(128,2,activation = 'relu' , padding = 'same')(UpSampling2D(size = (2,2))(d2))
    d3 = concatenate([s2,d3], axis = 3)
    d3 = Conv2D(128,3,activation = 'relu' , padding = 'same')(d3)
    d3 = Conv2D(128,3,activation = 'relu' , padding = 'same')(d3)

    d4 = Conv2D(64,2,activation = 'relu' , padding = 'same')(UpSampling2D(size = (2,2))(d3))
    d4 = concatenate([s1,d4], axis = 3)
    d4 = Conv2D(64,3,activation = 'relu' , padding = 'same')(d4)
    d4 = Conv2D(64,3,activation = 'relu' , padding = 'same')(d4)

    d4 = Conv2D(2,3,activation = 'relu' , padding = 'same')(d4)
    out = Conv2D(1,1,activation = 'sigmoid')(d4)
    
    model = Model(inputs,out)
    return model
    
    
 import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
import shutil
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy



model = unet_model()
# Define optimizer and loss
optimizer = Adam(learning_rate=0.001)
criterion = BinaryCrossentropy()
num_epochs = 2
model.compile(optimizer=optimizer, loss=criterion, metrics=['accuracy'])

#Training Code
for epoch in range(num_epochs):
    for batch_idx, (images, masks) in enumerate(trainDataloader):
        images = images.permute(0, 2, 3, 1)
        masks = masks.permute(0, 2, 3, 1)

        outputs = model(images)
        loss = criterion(outputs, masks)

        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(trainDataloader)}], Loss: {loss.numpy():.4f}')


import tensorflow as tf
import numpy as np


# Define Accuracy calculation function
def calculate_accuracy(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    correct_predictions = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), tf.float32))
    total_pixels = tf.size(y_true, out_type=tf.float32)
    return correct_predictions / total_pixels

#evaluation
def evaluate_model(model, dataloader):
    total_loss = 0
    total_iou = 0
    total_accuracy = 0
    num_batches = 0

    for batch_idx, (images, masks) in enumerate(dataloader):
        #adjust image and mask shape for model compatibility
        images = images.permute(0, 2, 3, 1)
        masks = masks.permute(0, 2, 3, 1)
   
        outputs = model(images, training=False)
        
        loss = criterion(masks, outputs)
        total_loss += loss.numpy()
        
        # alculate IoU and accuracy
        batch_iou = calculate_iou(masks, outputs)
        batch_accuracy = calculate_accuracy(masks, outputs)
        total_iou += batch_iou.numpy()
        total_accuracy += batch_accuracy.numpy()
        
        num_batches += 1
        print(f'Batch [{batch_idx+1}/{len(dataloader)}] - Loss: {loss.numpy():.4f}, Accuracy: {batch_accuracy.numpy():.4f}')

    
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    print(f"\nEvaluation Results - Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}")
    return avg_loss, avg_iou, avg_accuracy

# Run evaluation
evaluate_model(model, testDataLoader)


#######
#Evaluation Results - Avg Loss: 0.6795, Avg Accuracy: 0.0023
######