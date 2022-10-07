import pandas as pd
import random
import math
import os
import numpy as np

random.seed(0)

dataset = 'USD'
train_file = 'train_usd50.csv'
file_path = '../data/' + dataset + '/annotations/'
wlimages = np.concatenate(([5], np.linspace(10, 90, 9)))

# reading USD-50: randomly choosen 50% images
df = pd.read_csv(os.path.join(file_path, train_file))
# unique image names
udf = df['image'].unique()
nimg = udf.shape[0]

numbers = list(range(0, nimg))
for loc, wl in enumerate(wlimages):
    if loc == 0:
        idx = random.sample(numbers, math.ceil(nimg * wl / 100))
        wlprev = wl
    else:
        cidx = list(set(numbers) - set(idx))
        idx.extend(random.sample(cidx, math.ceil(nimg * (wl - wlprev) / 100)))
        wlprev = wl

    # well-labeled images
    ndf = df[df['image'].isin(udf[idx])]
    ndf.to_csv(os.path.join(file_path, dataset.lower() + '_wl' + str(int(wl)) + '.csv'), index=False)

    # well-labeled to point annotation (replace box with points)
    pidx = list(set(range(0, nimg)) - set(idx))
    ndf = df[df['image'].isin(udf[pidx])]

    ndf.to_csv(os.path.join(file_path, dataset.lower() + '_pt' + str(int(100 - wl)) + '_box.csv'), index=False)

    xc = 0.5 * (ndf.xmin.values + ndf.xmax.values)
    yc = 0.5 * (ndf.ymin.values + ndf.ymax.values)
    ndf = ndf.assign(xmin=xc)
    ndf = ndf.assign(xmax=xc)
    ndf = ndf.assign(ymin=yc)
    ndf = ndf.assign(ymax=yc)
    ndf.to_csv(os.path.join(file_path, dataset.lower() + '_pt' + str(int(100 - wl)) + '.csv'), index=False)
