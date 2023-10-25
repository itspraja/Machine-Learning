"""
Author: Rohit Prajapati
Description: A Convolutional AutoEncoder, with 3 Hidden Layers and an added Denoiser.

Brief Inormation on __main__
Step 1: Download & Preprocess PASCAL VOC 2007 Data
Step 2: Split Dataset in Desired Ratios
Step 3: Creating and Autoencoder with Latent Dimension 256 and Choosing the Best DataSplit
Step 4: Finding the Best Latent Dimension
Step 5: Reconstruction Errors for Every Autoencoder Model
Step 6: Evaluating Metrics and Reporting MSE and MAE
Step 7: Visualizing the Images (Original, Masked, Reconstructed)
Step 8: Choosing Another Metric to judge Image Quality: PSNR
"""

import cv2
import os
import wget
import tarfile
import numpy as np
import matplotlib.pyplot as plot
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape, Conv2DTranspose, UpSampling2D, Input, GaussianNoise
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError

# Pre-Processing and Normalizing the Images in the Dataset
def DLAndPreProcess(fixedHeight, fixedWidth):
    # PASCAL VOC 2007 Dataset Download
    url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"
    wget.download(url,"VOCtrainval_06-Nov-2007.tar")

    with tarfile.open("VOCtrainval_06-Nov-2007.tar", "r") as tar:
        tar.extractall()

    imageDir = "../VOCdevkit/VOC2007/JPEGImages"
    
    imageData = []
    imagePath = []
    
    for filename in os.listdir(imageDir):
        if filename.endswith('.jpg'):
            imagePath.append(os.path.join(imageDir, filename))
            
    for path in imagePath:
        image = cv2.imread(path)
        image = cv2.resize(image, (fixedHeight, fixedWidth))
        image = image / 255.0
        imageData.append(image)
    
    imageData = np.array(imageData)
    return imageData

# Data Splitting (Training/Validation/Testing)
def splitDataset(data, training, validation, testing):
    split = validation / (validation + testing)
    
    trainingSize = int(training * data.shape[0])
    validationSize = int(validation * data.shape[0])
    
    trainData = data[:trainingSize]
    validationData = data[trainingSize:(trainingSize + validationSize)]
    testData = data[(trainingSize + validationSize):]
    
    return trainData, validationData, testData

def addMask(data, maskRatio):
    mask = np.random.random(data.shape) > maskRatio
    maskedImages = data * mask
    return maskedImages

def createEncoder(fixedHeight, fixedWidth, dimension):
    imageData = Input(shape = (fixedHeight, fixedWidth) + (3, ))
    
    noisyImageData = GaussianNoise(0.2)(imageData)
    
    X = Conv2D(32, (3, 3), padding='same', activation='relu')(noisyImageData)
    X = MaxPooling2D((2, 2), padding = 'same')(X)
    X = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(X)
    encoder = MaxPooling2D((2, 2), padding = 'same')(X)
    
    X = Conv2D(64, (3,3 ), activation = 'relu', padding = 'same')(encoder)
    X = UpSampling2D((2, 2))(X)
    X = Conv2D(32, (3, 3), activation = 'relu', padding = 'same')(X)
    X = UpSampling2D((2, 2))(X)
    decoder = Conv2D(3, (3, 3), activation = 'sigmoid', padding = 'same')(X)
    
    encoder = Flatten()(encoder)
    encoder = Dense(dimension, activation = 'relu')(encoder)
    
    AutoEncoder = Model(imageData, decoder)
    AutoEncoder.compile(optimizer=Adam(learning_rate = 0.001), loss=losses.mean_squared_error)
    
    return AutoEncoder

def chooseBestSplit():
    print("Step 2: Splitting the Datasets into: 80/10/10 and 70/10/20")
    training80, validation80, testing80 = splitDataset(imageData, 0.8, 0.1, 0.1)
    training70, validation70, testing70 = splitDataset(imageData, 0.7, 0.1, 0.2)
    print("\nStep 2 (Completed)\n")

    reConErr80 = []
    reConErr70 = []

    AE80 = createEncoder(128, 128, 256)
    AE70 = createEncoder(128, 128, 256)

    AE80.compile(optimizer=Adam(learning_rate = 0.001), loss=losses.mean_squared_error)
    AE70.compile(optimizer=Adam(learning_rate = 0.001), loss=losses.mean_squared_error)

    epoch, batch = 5, 32

    AE80.fit(training80,
             training80, epochs = epoch,
             batch_size = batch,
             validation_data = (validation80, validation80))

    AE70.fit((training70),
             training70, epochs = epoch,
             batch_size = batch,
             validation_data = (validation70, validation70))
    
    reConErr80 = AE80.evaluate(testing80, testing80)
    reConErr70 = AE70.evaluate(testing70, testing70)
    
    print("\nStep 3: Creating and Autoencoder with Latent Dimension 256 and Choosing the Best DataSplit")
    if reConErr70 < reConErr80:
        print("\nDataset Split (70/10/20) is a Better Fit!")
        return training70, validation70, testing70
    else:
        print("\nDataset Split (80/10/10) is a Better Fit!")
        return training80, validation80, testing80

def evaluateLoss(dimension):
    epoch, batch = 5, 32
    
    AutoEncoder = createEncoder(128, 128, dimension)
    AutoEncoder.compile(optimizer=Adam(learning_rate = 0.001), loss=losses.mean_squared_error)
        
    AutoEncoder.fit(training,training, epochs = epoch,
                    batch_size = batch,
                    validation_data = (validation, validation))

    testLoss = AutoEncoder.evaluate(testing, testing)
    return testLoss

def evaluateDimensions(dimensions):
    bestModel = None
    lowTestLoss = float('inf')
    
    for dimension in dimensions:
        testLoss = evaluateLoss(dimension)
        if testLoss < lowTestLoss:
            lowTestLoss = testLoss
            bestModel = dimension

    return dimension

def evaluateMetrics(dimensions, masks):

    evalResult = "Models\n(Dimension, Mask)\n"
    allResults = []
    
    epoch, batch = 5, 32
    for dimension in dimensions:
        for mask in masks:
            AutoEncoder = createEncoder(128, 128, dimension)
            AutoEncoder.compile(optimizer=Adam(learning_rate = 0.001), loss=losses.mean_squared_error)
            AutoEncoder.fit(training,training, epochs = epoch,
                    batch_size = batch,
                    validation_data = (validation, validation))
            maskedTesting = addMask(testing, mask)
            reconImages = AutoEncoder.predict(maskedTesting)

            mse = MeanSquaredError()(testing, reconImages).numpy()
            mae = MeanAbsoluteError()(testing, reconImages).numpy()

            allResults.append([dimension, mask, mse, mae])
            evalResult += f"({dimension},{mask*100}%)\tMSE: {mse:.4f}\t | MAE: {mae:.4f}\n"
    
    npResults = np.array(allResults)
    return evalResult, npResults

def showImages():
    epoch, batch = 5, 32
    
    AutoEncoder = createEncoder(128, 128, 16)
    AutoEncoder.compile(optimizer=Adam(learning_rate = 0.001), loss=losses.mean_squared_error)
    maskedTraining = addMask(training, 0.2)
    AutoEncoder.fit(training,training, epochs = epoch,
                    batch_size = batch,
                    validation_data = (validation, validation))
    reconImages = AutoEncoder.predict(maskedTraining)
    
    n = 5
    plot.figure(figsize = (10,2))
    for i in range(n):
        ax = plot.subplot(1, n, i+1)
        plot.title("Original")
        plot.imshow(testing[i])
        plot.axis('off')
    plot.show()
    
    plot.figure(figsize = (10,2))
    for i in range(n):
        ax = plot.subplot(1, n, i+1)
        plot.title("Masked")
        plot.imshow(maskedTraining[i])
        plot.axis('off')

    plot.show()

    plot.figure(figsize = (10,2))
    for i in range(n):
        ax = plot.subplot(1, n, i+1)
        plot.title("Reconstructed")
        plot.imshow(reconImages[i])
        plot.axis('off')

    plot.show()

def evaluateReconErrors(dimensions, masks):
    reconErrors = []

    epoch, batch = 5, 32
    for dimension in dimensions:
        dimensionErrors = []
        for mask in masks:
            AutoEncoder = createEncoder(128, 128, dimension)
            AutoEncoder.compile(optimizer=Adam(learning_rate=0.001), loss=losses.mean_squared_error)

            AutoEncoder.fit(training, training, epochs=epoch, batch_size=batch, validation_data=(validation, validation))

            maskedTesting = addMask(testing, mask)
            reconImages = AutoEncoder.predict(maskedTesting)
            mse = MeanSquaredError()(testing, reconImages).numpy()
            
            dimensionErrors.append(mse)

        reconErrors.append(dimensionErrors)

    return reconErrors

if __name__ == '__main__':

    # Variables 
    dimensions = [256, 128, 64, 32, 16]
    masks = [0.2, 0.4, 0.6, 0.8]
    global training, validation, testing
    fixedHeight, fixedWidth = 128, 128
    # Step 1
    print("Step 1: Downloading and Preprocessing the Dataset")
    imageData = DLAndPreProcess(fixedHeight, fixedWidth)
    print("\nStep 1 (Completed)\n")
    
    # Step 2 & Step 3
    training, validation, testing = chooseBestSplit()
    print("\nStep 3 (Completed)\n")
    
    # Step 4
    print("\nStep 4: Finding the Best Dimension")
    bestDimension = evaluateDimensions(dimensions)
    print("\nBest Dimension: ", bestDimension)
    print("\nStep 4 (Completed)\n")
    MSE, MAE, outputString = evaluateMetrics(bestDimension, masks)
    
    # Step 5
    print("\nStep 5: Reconstruction Errors for Every Autoencoder Model")
    reconErrors = evaluateReconErrors(dimensions, masks)
    x, y = np.meshgrid(dimensions, masks)
    z = np.array(reconErrors).T
    
    plot.figure(figsize = (10, 6))
    heatmap = plot.pcolormesh(x, y, z, shading='auto', cmap='viridis')
    plot.colorbar(heatmap, label='Reconstruction Error (MSE)')
    plot.xlabel('Dimension')
    plot.ylabel('Mask')
    plot.title('Reconstruction Errors')
    plot.show()
    print("\nStep 5 (Completed)\n")
    
    # Step 6
    print("\nStep 6: Evaluation Metrics")
    outputString, allResults = evaluateMetrics(dimensions, masks)
    minMSE = np.argmin(allResults[:, 2])
    minMAE = np.argmin(allResults[:, 3])
     
    print("\nMetrics for All Models")
    print(outputString)
    print(f"""\nJudging the Quality of Autoencoders.\n
        [#]Priortizing Models that are better at Capturing Overall Trends and are not strongly affected by Outliers. MAE is the suitable choice.
        \tModel with the Lowest MAE makes a Viable Option.
        \nDimension: {int(allResults[minMAE,0])}, Mask: {allResults[minMAE,1]*100}% - MSE: {allResults[minMAE,2]:.4f} | MAE: {allResults[minMAE,3]:.4f}

        [#]Ensuring Model is Minimizing Errors on a Per-Pixel Basis and some Deviation is acceptable for a better reconstruction. MSE is the suitable choice.
        \tModel with the Lowest MSE makes a Viable Option.
        \nDimension: {int(allResults[minMSE,0])}, Mask: {allResults[minMSE,1]*100}% - MSE: {allResults[minMSE,2]:.4f} | MAE: {allResults[minMSE,3]:.4f}
        """)
    print("\nStep 6 (Completed)\n")
    
    # Step 7
    print("Step 7: Visual Comparing of the Images")
    showImages()
    print("\nStep 7 (Completed)\n")
    
    # Step 8
    print("Step 8: Another Metric: Peak Signal Noise Ration")
    print("\n(Dimension, Mask)\tPSNR Ratio")
    for index in range(allResults.shape[0]):
        print(f"({int(allResults[index,0])}, {allResults[index,1]*100}%)\t\t{allResults[index,4]:.4f}")
    
    maxPSNR = np.argmax(allResults[:, 4])
    print(f"""\nAnother Metric for Judging Image Quality: Peak Signal Noise Ratio          
          \n[#] A Higher Value of PSNR indicated better Image Quality. As we can see from the above values:
          \nDimension: {int(allResults[maxPSNR,0])}, Mask: {allResults[maxPSNR,1]*100}%\t\tPSNR: {allResults[maxPSNR,4]:.4f}
          """)
    print("\nStep 8 (Completed)\n")
    
    # Step 9
    print("Step 9: Saving the Model with Best Dimension")
    AutoEncoder = createEncoder(fixedHeight, fixedWidth, bestDimension)
    AutoEncoder.save('AutoEncoderModel.h5')
    print("\nStep 9 (Completed)\n")
# End of Code
