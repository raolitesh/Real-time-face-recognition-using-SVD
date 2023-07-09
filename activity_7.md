## Refer to activity 7 question
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
singval = np.array([183522.5272366 ,  89687.42645889,  48113.79566265,  45868.87578377,
        38004.94316869,  33753.34493299,  26919.89908823,  23421.21599414,
        22180.3064635 ,  20251.53488754,  18351.57568692,  18265.13695643,
        15712.38748689,  15226.24696913,  14647.97585729,  14515.38230552,
        13412.60712139,  13081.14716302,  12897.35882883,  12265.67173092,
        11440.42041148,  11161.23123572,  10819.40814407,  10672.53744009,
        10437.50520957,  10013.28959603,   9847.61019423,   9354.29430794,
         9094.8072678 ,   8903.91161079,   8766.76400986,   8410.01194593,
         8244.18820091,   8027.80631486,   7953.23549445,   7790.96075683,
         7616.71400524,   7483.03264883,   7284.86045722,   7209.02046036,
         7085.55191745,   7027.21321797,   6875.27260647,   6809.78217014,
         6683.45009236,   6470.43746008,   6449.40208239,   6261.04454744,
         6220.75740443,   6027.14889668,   5919.5189087 ,   5860.83612282,
         5815.89145752,   5693.71149421,   5615.99971158,   5551.82462013,
         5296.16915218,   5242.78868133,   5130.27219343,   5053.29351876,
         5011.2907792 ,   4934.91012378,   4913.17044003,   4771.35275575,
         4699.84398312,   4661.77587471,   4585.19438388,   4538.79845741,
         4383.2290089 ,   4349.89910457,   4259.30433819,   4229.69480385,
         4136.73608725,   4054.88584587,   4022.77568604,   3930.34336962,
         3869.49298735,   3835.68205436,   3753.04677513,   3713.08064843,
         3632.03078559,   3518.43701878,   3393.38777102,   3363.36518342,
         3238.94132047,   3178.94069446,   3100.76999926,   3009.95818279,
         2971.89114428,   2237.43139141])
```

```python
p = np.arange(1,91,1)
```

```python
%matplotlib notebook
plt.figure(figsize=(8,5))
plt.plot(p,singval)
plt.xlabel('P')
plt.ylabel('S (singular values)')
plt.title('Singular values as a function of P')
plt.plot(20,12600, 'o', markersize=8, color = 'red')
plt.annotate('The Optimal P Value is 20', xytext=(40,75000), xy=(21,15000), arrowprops={'facecolor': 'yellow'})
plt.grid(True)
#plt.savefig('q7.png', dpi=300, bbox_inches='tight')
plt.show()
```