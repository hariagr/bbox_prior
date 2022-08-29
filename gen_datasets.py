import pandas as pd
import random
import math
import os
import numpy as np

random.seed(0)

file_path = '../data/USD/annotations/'
pruning = 5

# create USD-50: randomly choose 50% images
df = pd.read_csv(os.path.join(file_path, 'train_pruned_50.csv'))

# unique image names
udf = df['image'].unique()
nimg = udf.shape[0]

idx = random.sample(range(0, nimg), math.ceil(nimg*pruning/100))
ndf = df[df['image'].isin(udf[idx])]

ndf.to_csv(os.path.join(file_path, 'train_pruned_' + str(pruning) + '.csv'), index=False)

