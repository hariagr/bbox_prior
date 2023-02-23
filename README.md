# Object Detection with Point Annotations

This is the PyTorch based code for the paper "Bounding Box Priors for Cell Detection with Point Annotations"{https://arxiv.org/abs/2211.06104}.

The paper proposes a weakly semi-supervised machine learning algorithm to detect objects with point annotations.

# Example code for training
!python train.py --train-file train.csv\
  --workers 16 --batch-size 2 --epoch 50 --lr 0.01 --beta 0.99 --eval-freq 50\
  --device cuda --amp --balance\
  --gt-bbox-loss l1 \
  --data-path ../data/BCCD/ --results-dir ../results --config bccd_wl100

# Bugs and Support
Please send an email at hariom85@gmail.com.
  


