import pandas as pd
import random
import math
import os
import numpy as np

random.seed(1)
np.random.seed(1)

file_path = '../data/USD/annotations/'
# reading USD-50: randomly choosen 50% images
df = pd.read_csv(os.path.join(file_path, 'train_usd50.csv'))

classes = ['leuko', 'eryth', 'epith', 'epithn', 'cryst', 'cast', 'mycete']

for cls in classes:
    ndf = df.loc[df['label'] == cls]
    filename = cls + '.csv'
    ndf.to_csv(os.path.join(file_path, filename), index=False)

    udf = ndf
    udf = udf.reset_index(drop=True)
    nlabels = udf.shape[0]

    wlimages = np.concatenate(([5], np.linspace(10, 90, 9)))
    numbers = list(range(0, nlabels))
    for loc, wl in enumerate(wlimages):
        if loc == 0:
            idx = random.sample(numbers, math.ceil(nlabels * wl / 100))
            wlprev = wl
        else:
            cidx = list(set(numbers) - set(idx))
            idx.extend(random.sample(cidx, math.ceil(nlabels * (wl - wlprev) / 100)))
            wlprev = wl

        # well-labeled images
        ndf = udf.iloc[idx]
        ndf.to_csv(os.path.join(file_path, str(cls) + '_wl' + str(int(wl)) + '.csv'), index=False)

        # well-labeled to point annotation (replace box with points)
        pidx = list(set(range(0, nlabels)) - set(idx))
        ndf = udf.iloc[pidx]
        ndf.to_csv(os.path.join(file_path, str(cls) + '_pt' + str(int(100 - wl)) + '_box.csv'), index=False)

        area = np.sqrt((ndf['xmax'] - ndf['xmin']) * (ndf['ymax'] - ndf['ymin']))

        # generate points
        xc = 0.5 * (ndf.xmin.values + ndf.xmax.values)
        yc = 0.5 * (ndf.ymin.values + ndf.ymax.values)
        ndf = ndf.assign(xmin=xc)
        ndf = ndf.assign(xmax=xc)
        ndf = ndf.assign(ymin=yc)
        ndf = ndf.assign(ymax=yc)
        ndf.to_csv(os.path.join(file_path, str(cls) + '_pt' + str(int(100 - wl)) + '.csv'), index=False)

        # add noise
        area[area < 18] = 18
        area[area > 198] = 198
        linear_mean = 5 + ((17 - 5) / (198 - 18)) * (area - 18)
        std = 3
        e = np.random.normal(0, 1, linear_mean.size)
        e[e < -1.0] = -1.0
        e[e > 1.0] = 1.0  # more than one std. dev. is practically unrealistic
        e = linear_mean + std * e
        e[e < 0] = 0.0  # distance can't be negative
        theta = np.random.uniform(0, 2 * np.pi, linear_mean.size)
        xc = xc + e * np.cos(theta)
        yc = yc + e * np.sin(theta)
        ndf = ndf.assign(xmin=xc)
        ndf = ndf.assign(xmax=xc)
        ndf = ndf.assign(ymin=yc)
        ndf = ndf.assign(ymax=yc)
        ndf.to_csv(os.path.join(file_path, str(cls) + '_pt' + str(int(100 - wl)) + '_noisy.csv'), index=False)

if 0:
    cls = "cast"
    rows = df.loc[df['label'] == cls]
    udf = rows["image"].unique()
    ndf = df[df['image'].isin(udf)]

    filename = 'train_' + cls + '.csv'
    ndf.to_csv(os.path.join(file_path, filename), index=False)

    # unique image names
    df = ndf
    udf = df['image'].unique()
    nimg = udf.shape[0]

    wlimages = np.concatenate(([5], np.linspace(10, 90, 9)))
    for wl in wlimages:
        idx = random.sample(range(0, nimg), math.ceil(nimg*wl/100))

        # well-labeled images
        ndf = df[df['image'].isin(udf[idx])]
        ndf.to_csv(os.path.join(file_path, str(cls) + '_wl' + str(int(wl)) + '.csv'), index=False)

        # well-labeled to point annotation (replace box with points)
        idx = list(set(range(0, nimg)) - set(idx))
        ndf = df[df['image'].isin(udf[idx])]

        xc = 0.5 * (ndf.xmin.values + ndf.xmax.values)
        yc = 0.5 * (ndf.ymin.values + ndf.ymax.values)
        ndf = ndf.assign(xmin=xc)
        ndf = ndf.assign(xmax=xc)
        ndf = ndf.assign(ymin=yc)
        ndf = ndf.assign(ymax=yc)
        ndf.to_csv(os.path.join(file_path, str(cls) + '_pt' + str(int(100 - wl)) + '.csv'), index=False)