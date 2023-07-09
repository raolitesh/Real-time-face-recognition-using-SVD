## Refer to activity 1 question

```python
from PIL import Image #importing the Image function to read images
import matplotlib.pyplot as plt #import matplotlib for plotting of images
import os #import os to access images in the folder/file
import numpy as np
import pandas as pd
from simple_colors import *

```
```python
path = 'data' #all images are in the folder named data
im = os.listdir(path) #accessing the images in folder
print(blue('Loading all 165 images',['bold', 'reverse']))
for img in im:
    x = Image.open(os.path.join(path,img))
    print('title:' + '\t' + str(img) + '\t'+ '\t' + str(x))
```

```python
#plotting 25 faces from these 165 images
img_data = []
path = 'data' #all images are in the folder named data
im = os.listdir(path)
for img in im:
    x = Image.open(os.path.join(path,img))
    img_data.append(x)

print(blue('Printing first 25 images',['bold', 'reverse']))
img_data1 = img_data[0:25]   
plt.figure(figsize=(8,6))
for i in range(len(img_data1)):
    plt.style.use('default')
    plt.subplot(5,5,i+1)
    plt.imshow(img_data1[i],cmap='Greys_r')
    #plt.savefig('q1.png')
    plt.axis('off')
```
