import pandas as pd
import random
import math
import os
import numpy as np

random.seed(0)

file_path = '../data/USD/annotations/'
wlimages = np.concatenate(([5], np.linspace(10, 90, 9)))

# reading USD-50: randomly choosen 50% images
df = pd.read_csv(os.path.join(file_path, 'train_usd50.csv'))
# unique image names
udf = df['image'].unique()
nimg = udf.shape[0]

## under-representation (work with only single class)
## show atleast 10-20 images with bounding boxes for a particular class

for wl in wlimages:
    idx = random.sample(range(0, nimg), math.ceil(nimg*wl/100))

    # well-labeled images
    ndf = df[df['image'].isin(udf[idx])]
    ndf.to_csv(os.path.join(file_path, 'train_usd50_wl' + str(int(wl)) + '.csv'), index=False)

    # well-labeled to point annotation (replace box with points)
    idx = list(set(range(0,nimg)) - set(idx))
    ndf = df[df['image'].isin(udf[idx])]

    xc = 0.5 * (ndf.xmin.values + ndf.xmax.values)
    yc = 0.5 * (ndf.ymin.values + ndf.ymax.values)
    ndf = ndf.assign(xmin=xc)
    ndf = ndf.assign(xmax=xc)
    ndf = ndf.assign(ymin=yc)
    ndf = ndf.assign(ymax=yc)
    ndf.to_csv(os.path.join(file_path, 'train_usd50_pt' + str(int(100 - wl)) + '.csv'), index=False)
