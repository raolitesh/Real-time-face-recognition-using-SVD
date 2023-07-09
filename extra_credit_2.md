## Refer to extra credit question 3
```python
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from simple_colors import *
import pandas as pd
from numpy import asarray
import time
```
```python
np.seterr(divide='ignore', invalid='ignore')
```

```python
def query(face, database):
    
    # creating a database
    matrixdata1 = []
    images = os.listdir(database)
    for i in images:
        x = Image.open(os.path.join(database,i))
        predata = np.asarray(x)
        shape = predata.shape
        data1 = predata.ravel()
        reshapedata = data1.T
        matrixdata1.append(reshapedata)
    A = np.array(matrixdata1).T # this is matrix of images in database
    #print('matrix of m x n images:'  ,matrixdata2)  # we can check the values of matrix if we need
    #print('Dimension of m x n matrix'  ,matrixdata2.shape) # we can check the dimension of the matrix
    
    # calculating the mean face

    mean = np.zeros((A.shape[0],1))
    for i in range(0,A.shape[0]):
        mean[i,:] = (sum(A[i,:]))/A.shape[1]
        
    # calculating norm of the mean
    #normean = np.linalg.norm(mean)
    #print('norm of the mean')
    #print(normean)
    #print()
    #print(blue('The shape of the mean of dataset',['bold','reverse']))
    #print(mean.shape)
    #print()

    # normalizing the dataset

    B = np.zeros((A.shape[0],A.shape[1]))
    for j in range(0,A.shape[1]):
        B[:,j] = A[:,j] - mean[:,0]

    #print(green('normalized matrix of m x n images:',['bold','reverse']))
    #print(B)
    #print()
    #print(green('Dimension of normalized m x n matrix',['bold','reverse']))
    #print(B.shape)
    
    # executing SVD using python inbuilt function

    U, Sigma, Vt = np.linalg.svd(B, full_matrices=False)
    
    #print('U matrix=')
    #print(U)
    
    a = B.shape[0]
    b = B.shape[1]
    
    # selecting first p=20 columns of U
    p = 20
    U1 = U[:,0:p]
    #print()
    #print('U1:')
    #print(U1.shape)
    
    # projecting the data F on reduced face space
    data2 = np.zeros((p,b))
    data2 = U1.T.dot(B) # this gives matrix of x coordinates as per equation 15 in the paper
    #print()
    #print('Project Face database')
    #print(data2)
    
    
    #creating a face for identification
    matrixface1 = []
    image = Image.open(face) 
    preface = np.asarray(image)
    face1 = preface.ravel()
    reshapeface = face1.T
    matrixface1.append(reshapeface)
    matrixface2 = np.array(matrixface1).T
    
        
    #print('matrix of m x 1 image:', matrixface2) # we can check the values of matrix if we need
    #print('Dimension of m x 1 matrix', matrixface2.shape) # we can check the dimension of the matrix
    
    # normalizing it 

    B2 = np.zeros((matrixface2.shape[0],matrixface2.shape[1]))
    for j in range(0,matrixface2.shape[1]):
        B2[:,j] = matrixface2[:,j] - mean[:,0]
        
    #print('The normalized face in Query database')
    #print(B2)
    
    
    # projecting the data Q on reduced face space
    a2 = B2.shape[0]
    b2 = B2.shape[1]
    data3 = np.zeros((p,b2))
    data3 = U1.T.dot(B2) # this gives matrix of x coordinates as per equation 15 in the paper
    #print()
    #print('Project Face database')
    #print(data3)
    
    # comparing the face in query with the images in database
    tol = 3000 # arbitrary chosen tolerance level. We can change it to see changes in results.
    
    st = time.process_time()

    error = np.zeros(data2.shape[1])
    for j in range(0,data2.shape[1]):
        error[j] = np.linalg.norm(data3[:,0] - data2[:,j])
        
    
    et = time.process_time()
    res = et - st
    print('CPU Execution time:', res, 'seconds')
    print()
    #print(error)
    #print(error.shape)
    
    # implementing the algorithm for face f to be defined as classified as face fi when the minimum error (i) is less than some
# predefined threshold error (0). Otherwise the face f is classified as unknown face", and optionally be used to initialize a new identity.
    
    m = min(error)
    #print(m)
    
    o = 0
    
    for i in range(0,data2.shape[1]):
        if np.all(error > tol):
            print(red('This subject:',['bold','reverse']), face, red(':is an unknown face and hence will be used to initialize a new identity',['bold', 'reverse']))
            break
        else:
            if error[i] < tol and error[i] == m:
                o = i
                print()
                #print('The closest match to the query subject:', face, ':is known subject', o+1)
        
                #print(green('The image of the to be identified query subject and closely identified match is below',['reverse']))
                
                print(green('The closest match to the query',['bold', 'reverse']), face, 
                         green('is', ['bold','reverse']), images[o])
                print()
                print(green('and its image number in the database is',['bold','reverse']), o+1)
                
                imgdata = np.asarray(A[:,o]).reshape(243,320)
                image = Image.fromarray(imgdata)   

                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 8))
                fig.tight_layout() 
                plt.subplot(1,2,2)
                plt.imshow(image,cmap='Greys_r')
                plt.title('Closest match')
                plt.axis('off')
                plt.subplot(1,2,1)
                img = Image.open(face)
                plt.imshow(img,cmap='Greys_r')
                plt.title('To be identified subject')
                plt.axis('off')
                #plt.savefig('q4.png')
                plt.show()
```
```python
# comparing faces in unknown3 folder with faces in known3 folder
for img in os.listdir('unknown3'):
    query(os.path.join('unknown3',img), 'known3')
    
```
```python
def query(face, database):
    
    # creating a database
    matrixdata1 = []
    images = os.listdir(database)
    for i in images:
        x = Image.open(os.path.join(database,i))
        predata = np.asarray(x)
        shape = predata.shape
        data1 = predata.ravel()
        reshapedata = data1.T
        matrixdata1.append(reshapedata)
    A = np.array(matrixdata1).T # this is matrix of images in database
    #print('matrix of m x n images:'  ,matrixdata2)  # we can check the values of matrix if we need
    #print('Dimension of m x n matrix'  ,matrixdata2.shape) # we can check the dimension of the matrix
    
    # calculating the mean face

    mean = np.zeros((A.shape[0],1))
    for i in range(0,A.shape[0]):
        mean[i,:] = (sum(A[i,:]))/A.shape[1]
        
    # calculating norm of the mean
    #normean = np.linalg.norm(mean)
    #print('norm of the mean')
    #print(normean)
    #print()
    #print(blue('The shape of the mean of dataset',['bold','reverse']))
    #print(mean.shape)
    #print()

    # normalizing the dataset

    B = np.zeros((A.shape[0],A.shape[1]))
    for j in range(0,A.shape[1]):
        B[:,j] = A[:,j] - mean[:,0]

    #print(green('normalized matrix of m x n images:',['bold','reverse']))
    #print(B)
    #print()
    #print(green('Dimension of normalized m x n matrix',['bold','reverse']))
    #print(B.shape)
    
    # executing SVD using python inbuilt function

    U, Sigma, Vt = np.linalg.svd(B, full_matrices=False)
    
    #print('U matrix=')
    #print(U)
    
    a = B.shape[0]
    b = B.shape[1]
    
    # selecting first p=20 columns of U
    p = 20
    U1 = U[:,0:p]
    #print()
    #print('U1:')
    #print(U1.shape)
    
    # projecting the data F on reduced face space
    data2 = np.zeros((p,b))
    data2 = U1.T.dot(B) # this gives matrix of x coordinates as per equation 15 in the paper
    #print()
    #print('Project Face database')
    #print(data2)
    
    
    #creating a face for identification
    matrixface1 = []
    image = Image.open(face) 
    preface = np.asarray(image)
    face1 = preface.ravel()
    reshapeface = face1.T
    matrixface1.append(reshapeface)
    matrixface2 = np.array(matrixface1).T
    
        
    #print('matrix of m x 1 image:', matrixface2) # we can check the values of matrix if we need
    #print('Dimension of m x 1 matrix', matrixface2.shape) # we can check the dimension of the matrix
    
    # normalizing it 

    B2 = np.zeros((matrixface2.shape[0],matrixface2.shape[1]))
    for j in range(0,matrixface2.shape[1]):
        B2[:,j] = matrixface2[:,j] - mean[:,0]
        
    #print('The normalized face in Query database')
    #print(B2)
    
    
    # projecting the data Q on reduced face space
    a2 = B2.shape[0]
    b2 = B2.shape[1]
    data3 = np.zeros((p,b2))
    data3 = U1.T.dot(B2) # this gives matrix of x coordinates as per equation 15 in the paper
    #print()
    #print('Project Face database')
    #print(data3)
    
    # comparing the face in query with the images in database
    tol = 1000 # arbitrary chosen tolerance level. We can change it to see changes in results.
    
    st = time.process_time()

    error = np.zeros(data2.shape[1])
    for j in range(0,data2.shape[1]):
        error[j] = np.linalg.norm(data3[:,0] - data2[:,j])
        
    
    et = time.process_time()
    res = et - st
    print('CPU Execution time:', res, 'seconds')
    print()
    #print(error)
    #print(error.shape)
    
    # implementing the algorithm for face f to be defined as classified as face fi when the minimum error (i) is less than some
# predefined threshold error (0). Otherwise the face f is classified as unknown face", and optionally be used to initialize a new identity.
    
    m = min(error)
    #print(m)
    
    o = 0
    
    for i in range(0,data2.shape[1]):
        if np.all(error > tol):
            print(red('This subject:',['bold','reverse']), face, red(':is an unknown face and hence will be used to initialize a new identity',['bold', 'reverse']))
            break
        else:
            if error[i] < tol and error[i] == m:
                o = i
                print()
                #print('The closest match to the query subject:', face, ':is known subject', o+1)
        
                #print(green('The image of the to be identified query subject and closely identified match is below',['reverse']))
                
                print(green('The closest match to the query',['bold', 'reverse']), face, 
                         green('is', ['bold','reverse']), images[o])
                print()
                print(green('and its image number in the database is',['bold','reverse']), o+1)
                
                imgdata = np.asarray(A[:,o]).reshape(243,320)
                image = Image.fromarray(imgdata)   

                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 8))
                fig.tight_layout() 
                plt.subplot(1,2,2)
                plt.imshow(image,cmap='Greys_r')
                plt.title('Closest match')
                plt.axis('off')
                plt.subplot(1,2,1)
                img = Image.open(face)
                plt.imshow(img,cmap='Greys_r')
                plt.title('To be identified subject')
                plt.axis('off')
                #plt.savefig('q4.png')
                plt.show()
        
   
```

```python
# comparing faces in unknown3 folder with faces in known3 folder
for img in os.listdir('unknown3'):
    query(os.path.join('unknown3',img), 'known3')
```
```python
def query(face, database):
    
    # creating a database
    matrixdata1 = []
    images = os.listdir(database)
    for i in images:
        x = Image.open(os.path.join(database,i))
        predata = np.asarray(x)
        shape = predata.shape
        data1 = predata.ravel()
        reshapedata = data1.T
        matrixdata1.append(reshapedata)
    A = np.array(matrixdata1).T # this is matrix of images in database
    #print('matrix of m x n images:'  ,matrixdata2)  # we can check the values of matrix if we need
    #print('Dimension of m x n matrix'  ,matrixdata2.shape) # we can check the dimension of the matrix
    
    # calculating the mean face

    mean = np.zeros((A.shape[0],1))
    for i in range(0,A.shape[0]):
        mean[i,:] = (sum(A[i,:]))/A.shape[1]
        
    # calculating norm of the mean
    #normean = np.linalg.norm(mean)
    #print('norm of the mean')
    #print(normean)
    #print()
    #print(blue('The shape of the mean of dataset',['bold','reverse']))
    #print(mean.shape)
    #print()

    # normalizing the dataset

    B = np.zeros((A.shape[0],A.shape[1]))
    for j in range(0,A.shape[1]):
        B[:,j] = A[:,j] - mean[:,0]

    #print(green('normalized matrix of m x n images:',['bold','reverse']))
    #print(B)
    #print()
    #print(green('Dimension of normalized m x n matrix',['bold','reverse']))
    #print(B.shape)
    
    # executing SVD using python inbuilt function

    U, Sigma, Vt = np.linalg.svd(B, full_matrices=False)
    
    #print('U matrix=')
    #print(U)
    
    a = B.shape[0]
    b = B.shape[1]
    
    # selecting first p=20 columns of U
    p = 20
    U1 = U[:,0:p]
    #print()
    #print('U1:')
    #print(U1.shape)
    
    # projecting the data F on reduced face space
    data2 = np.zeros((p,b))
    data2 = U1.T.dot(B) # this gives matrix of x coordinates as per equation 15 in the paper
    #print()
    #print('Project Face database')
    #print(data2)
    
    
    #creating a face for identification
    matrixface1 = []
    image = Image.open(face) 
    preface = np.asarray(image)
    face1 = preface.ravel()
    reshapeface = face1.T
    matrixface1.append(reshapeface)
    matrixface2 = np.array(matrixface1).T
    
        
    #print('matrix of m x 1 image:', matrixface2) # we can check the values of matrix if we need
    #print('Dimension of m x 1 matrix', matrixface2.shape) # we can check the dimension of the matrix
    
    # normalizing it 

    B2 = np.zeros((matrixface2.shape[0],matrixface2.shape[1]))
    for j in range(0,matrixface2.shape[1]):
        B2[:,j] = matrixface2[:,j] - mean[:,0]
        
    #print('The normalized face in Query database')
    #print(B2)
    
    
    # projecting the data Q on reduced face space
    a2 = B2.shape[0]
    b2 = B2.shape[1]
    data3 = np.zeros((p,b2))
    data3 = U1.T.dot(B2) # this gives matrix of x coordinates as per equation 15 in the paper
    #print()
    #print('Project Face database')
    #print(data3)
    
    # comparing the face in query with the images in database
    tol = 10000 # arbitrary chosen tolerance level. We can change it to see changes in results.
    
    st = time.process_time()

    error = np.zeros(data2.shape[1])
    for j in range(0,data2.shape[1]):
        error[j] = np.linalg.norm(data3[:,0] - data2[:,j])
        
    
    et = time.process_time()
    res = et - st
    print('CPU Execution time:', res, 'seconds')
    print()
    #print(error)
    #print(error.shape)
    
    # implementing the algorithm for face f to be defined as classified as face fi when the minimum error (i) is less than some
# predefined threshold error (0). Otherwise the face f is classified as unknown face", and optionally be used to initialize a new identity.
    
    m = min(error)
    #print(m)
    
    o = 0
    
    for i in range(0,data2.shape[1]):
        if np.all(error > tol):
            print(red('This subject:',['bold','reverse']), face, red(':is an unknown face and hence will be used to initialize a new identity',['bold', 'reverse']))
            break
        else:
            if error[i] < tol and error[i] == m:
                o = i
                print()
                #print('The closest match to the query subject:', face, ':is known subject', o+1)
        
                #print(green('The image of the to be identified query subject and closely identified match is below',['reverse']))
                
                print(green('The closest match to the query',['bold', 'reverse']), face, 
                         green('is', ['bold','reverse']), images[o])
                print()
                print(green('and its image number in the database is',['bold','reverse']), o+1)
                
                imgdata = np.asarray(A[:,o]).reshape(243,320)
                image = Image.fromarray(imgdata)   

                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 8))
                fig.tight_layout() 
                plt.subplot(1,2,2)
                plt.imshow(image,cmap='Greys_r')
                plt.title('Closest match')
                plt.axis('off')
                plt.subplot(1,2,1)
                img = Image.open(face)
                plt.imshow(img,cmap='Greys_r')
                plt.title('To be identified subject')
                plt.axis('off')
                #plt.savefig('q4.png')
                plt.show()
```

```python
# comparing faces in unknown3 folder with faces in known3 folder
for img in os.listdir('unknown3'):
    query(os.path.join('unknown3',img), 'known3')
```

