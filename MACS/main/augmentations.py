import numpy as np
import torch



def transform_train1(sample):
    
    # weak_aug = permutation(sample, max_segments=config.augmentation.max_seg)
    weak_aug =  jitter(sample, 0.001)

    return weak_aug
def transform_train2(sample):
    """Weak and strong augmentations"""
    weak_aug = scaling(sample, 0.001)

    return weak_aug



def jitter(x, sigma=0.8):

    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.1):
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)


