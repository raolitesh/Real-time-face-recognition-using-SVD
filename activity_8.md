# Activity 8


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
    
    # executing SVD

    # calculating A^t.A for matrix V
    AtA = B.T.dot(B)
    prod = AtA
    #print('The dimension of A^t.A is:', prod.shape)
    #print()

    #calculating eigenvalues and eigenvectors
    val1, vec1 = np.linalg.eig(prod)

    #sorting eigenvalues and their corresponding eigenvectors
    idx1 = val1.argsort()[::-1] 
    val1 = val1[idx1]
    vec1 = vec1[:,idx1]

    # matrix V and V^t
    V = vec1 # V matrix
    Vt = V.T # V^t
    #print('V^t matrix=')
    #print(Vt)

    #calculating singular values
    
    a = B.shape[0]
    b = B.shape[1]
    Sigma = np.zeros((a,b))

    for i in range(0,b): 
        for j in range(0,a):
            if i == j:
                Sigma[i,j] = np.sqrt(val1[i])
        
    #print('Sigma matrix=')
    #print(np.around(Sigma, decimals = 3))

    # calculating U matrix
    U = np.zeros((a,a))
    for i in range(0,b):
        U[:,i] = B.dot(V[:,i])/np.sqrt(val1[i])
    
    #print('U matrix=')
    #print(U)
    
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
    
    # comparing the images in query database with the images in face database
    
    st = time.process_time()
    
    error = []
    for j in range(0,data2.shape[1]):
        a = np.linalg.norm(data3[:,0] - data2[:,j])
        error.append(a)
        
    et = time.process_time()
    res = et - st
    print()
    print(red('CPU Execution time:',['bold']), res, 'seconds')

    #rint(magenta('The class of norm error is',['reverse']),type(error))
    #rint()
    #rint(blue('norm of the difference between vectors of images is given below',['reverse']))
    #rint()
    #rint(error)
    #rint()
    #rint(yellow('The number of images in our known database is:',['reverse'] ),len(error))
    #rint()
    #print(red('The closest match is subject:',['reverse']), error.index(min(error))+1)
    
    #plotting the unknown image and the closest match
    
    print()
    #print(green('The closest match to the query',['bold', 'reverse']), face, green('and closely identified match is below',['bold', 'reverse']) )
    
    print(green('The closest match to the query',['bold', 'reverse']), face, 
         green('is', ['bold','reverse']), images[error.index(min(error))])
    print()
    print(green('and its image number in the database is',['bold','reverse']), error.index(min(error))+1)
    
    imgdata = np.asarray(A[:,error.index(min(error))]).reshape(243,320)
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
    plt.title('Unknown subject')
    plt.axis('off')
    #plt.savefig('q4.png')
    plt.show()
    
```

```python

query('unknown/subject01.normal.gif', 'known')

```

```python
# comparing faces in unknown folder with faces in known folder

for img in os.listdir('unknown'):
    query(os.path.join('unknown',img), 'known')

```

```python
# comparing faces in unknown folder with faces in known folder

for img in os.listdir('unknown1'):
    query(os.path.join('unknown1',img), 'known1')
```

```python
# comparing faces in unknown folder with faces in known folder

for img in os.listdir('unknown2'):
    query(os.path.join('unknown2',img), 'known2')
```
```python
# comparing faces in unknown3 folder with faces in known3 folder

for img in os.listdir('unknown3'):
    query(os.path.join('unknown3',img), 'known3')

```

