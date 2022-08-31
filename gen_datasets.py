import pandas as pd
import random
import math
import os
import numpy as np

random.seed(0)

file_path = '../data/USD/annotations/'
wlimages = np.concatenate(([5], np.linspace(10, 90, 9)))

# create USD-50: randomly choose 50% images
df = pd.read_csv(os.path.join(file_path, 'train_usd50.csv'))
# unique image names
udf = df['image'].unique()
nimg = udf.shape[0]

for wl in wlimages:
    idx = random.sample(range(0, nimg), math.ceil(nimg*wl/100))
    ndf = df[df['image'].isin(udf[idx])]
    ndf.to_csv(os.path.join(file_path, 'train_usd50_wl' + str(int(wl)) + '.csv'), index=False)

