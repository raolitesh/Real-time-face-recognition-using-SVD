Note: 1) For all activities to download/create files or folders, use the same directory as Jupyter notebook is currently running in. Use the same names of files/folders as mentioned below in the relevant headings.

	2) Do not delete/change/replace any file or folder or their name until all activities are completed. Just follow the below steps sequentially.

	3) Some of the commands have commented out with '#' symbol in front to prevent unnecessary execution. But they were required initially to confirm the output. If you think, you need to check something, please delete the '#' symbol and execute the codes. But in a typical case, it is not required. 

The first command below needs to executed only once in the first file. Then, no need to execute them again.

1) 	Install the following libraries:
	!pip install Pillow
	!pip install simple_colors
	!pip install python-time

Execute codes in two bullet points below in all jupyter notebooks. Then follow activity specific task lists.

2) 	Import following packages:
	from PIL import Image
	import matplotlib.pyplot as plt
	import os
	import numpy as np
	from simple_colors import *
	import pandas as pd
	from numpy import asarray
	import time
	
3) Write this to ignore warning
	np.seterr(divide='ignore', invalid='ignore')

Follow activity specific reading list:

For activity 1:
	a) download the data (Yale database), unzip it and put all the images in the new folder and name the folder <data>
	b) execute line by line.

For activity 2:
	a) for img2vec function, use <data/filename>. For example, img2vec('data/subject15.glasses'). You can choose different file name from data folder instead of subject15.glasses
	b) for vec2img function, since all images have converted to vectors in a matrix <matrix2>. To retrieve the image back from vec, use <matrix2[:, column no.]>. For example, vec2img(matrix2[:,5]). You can choose any column you want the image of. 
	c) execute line by line

For activity 3:
	a) Create two folders and name them <unknown> and <known> respectively.
	b) execute line by line

For activity 4:
	a) For function query, use <query(foldername2/file name, foldername1)>. For example, query('unknown/subject01.normal.gif', 'known'). Foldername 2 is 'unknown' and foldername 1 is 'known'.  You can choose any file from the foldername 2 instead of <subject01.normal.gif>. But database must remain same as <known>
	b) execute line by line

For activity 5:
	a) Create six folders and name them known1, known2, known3, unknown1, unknown2, unknown3.
	b) execute line by line
	c) Wait for sometime for query to run. It may take sometime.

For activity 6:
	a) The path = 'known3' is used for SVD. You can choose any of the path from 'known', 'known1', 'known2'. Just replace it.  
	b) execute line by line

For activity 7:
	a) We have copied singular values for known3 database. You can change it any other database like to 'known', 'known1', 'known2'. For example, if you want singular values for 'known2'. Just run the activity 6 with path = 'known2'. Uncomment the code <print(np.around(Sigma, decimals = 3))>. Copy singular values from the output, and paste singular values in activity 7 notebook.
	b) Type <%matplotlib notebook> to have interactive plot.

For activity 8:
	a) We can check one image or we can check all images at once. For one image we have used <query('unknown/subject01.normal.gif', 'known')>. The syntax is <query('foldername2/filename from foldername2.gif', 'foldername1'). We can choose any foldername2 from the options 'unknown', 'unknown1', 'unknown2', 'unknown3'. And can choose any one file from one of these foldername2. Foldername1 we can choose from 'known', 'known1', 'known2', 'known3'. The choice should be made correspondingly. For example unknown1 goes with known1. We should not mix these.
	b) We can compare all images at once also. We have used <for img in os.listdir('unknown'):
    query(os.path.join('unknown',img), 'known')>. The syntax is <for img in os.listdir('foldername2'):
    query(os.path.join('foldername2',img), 'foldername1')>. Again, folder name should be chosen correspondingly as mentioned in point a) above.
	c) execute line by line
	d) Wait for sometime for query to run. It may take a while.

For extra credit 1:
	a) Follow the precautions for not mixing the folder names as mentioned for activity 8. We have used <query('unknown/subject01.normal.gif', 'known')>. The syntax is <query('foldername2/file name in folder name2.gif', 'foldername 1')>. We can choose any name for folder name 1 and folder name 2 subject to the precautions.
	b) execute line by line
	c) Wait for sometime for query to run. It may take a while.

For extra credit 3:
	a) Follow the precautions for not mixing the folder names as mentioned for activity 8. We have used <for img in os.listdir('unknown3'):
    query(os.path.join('unknown3',img), 'known3')>. The syntax is <for img in os.listdir('folder name 2'):
    query(os.path.join('folder name2',img), 'folder name 1')>. We can choose any name for folder name 1 and folder name 2 subject to the precautions.
	b) execute line by line
	c) Wait for sometime for query to run. It may take a while.
	








