import torch


def cal_tnorm_weights(model, dataloader, device, scaler=None):

    model.head.regression_head.cal_tnorm_weights = True
    model.train()
    for images, targets in dataloader:
        try:
            images = list(image.to(device, non_blocking=True) for image in images)
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                model(images, targets)
        except Exception as e:
            print(e)
            continue

    # Naive approach (may fails for a very large number of objects in the dataset)
    # you can use validation dataset to compute normalization weights
    target_normalization = model.head.regression_head.target_normalization
    weights = 1 / torch.sqrt(target_normalization['x2']/target_normalization['num'] - torch.pow(target_normalization['x']/target_normalization['num'], 2))
    #weights = weights / sum(weights)
    weights = tuple(weights.cpu().numpy())  # (0.29, 0.29, 0.20, 0.20)

    model.box_coder.weights = weights  # used while decoding boxes for prediction
    model.head.regression_head.box_coder.weights = weights  # used while encoding boxes for training
    model.head.regression_head.cal_tnorm_weights = False
    print(f"weights for normalizing targets are {weights}")
    print(f"std. dev. of targets are {1/torch.as_tensor(weights)}")

    return model
