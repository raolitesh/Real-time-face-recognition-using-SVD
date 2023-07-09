## Refer to activity 3 question
```python
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from simple_colors import *
import pandas as pd
from numpy import asarray
```
```python
# Dividing the dataset
# dataset unknown contains normal, happy, noglasses, rightlight, leftlight
# dataset known contains all other images

path = 'data'
images = os.listdir(path)
for i in images:
    x = Image.open(os.path.join(path,i))
    if i.endswith(('.rightlight', '.normal', '.leftlight', '.happy', '.noglasses')):
        x.save('./unknown/{}.gif'.format(i))
    else:
        x.save('./known/{}.gif'.format(i))
```
