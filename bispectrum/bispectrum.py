import torch
import numpy as np
import itertools


def translate_2d(img, v=0, h=0):
    """
    Given a (batch of) images and offset v, h returns cyclically translated images up in v and right in h.
    """
    if len(img.shape) < 3:
        img = np.expand_dims(img, 0)
    new_img = np.zeros_like(img)
    for i in range(img.shape[1]):
        for j in range(img.shape[2]):
            oldi = (i - v) % img.shape[1]
            oldj = (j - h) % img.shape[2]
            new_img[:, i, j] = img[:, oldi, oldj]
    return new_img

def bispectrum_2d(x):
    if len(x.shape) != 3:
        x = torch.unsqueeze(x, 0)
    m, n = x.shape[-2:]
    coords = list(itertools.product(range(m), range(n)))
    X = torch.fft.fft2(x)
    t = torch.stack([torch.roll(X, (m - idxs[0], n - idxs[1]), dims=(1, 2)) for idxs in coords])
    t = torch.swapaxes(t, 0, 1)
    t = t.reshape((t.shape[0], t.shape[1], -1))
    X = X.reshape((X.shape[0], -1))
    X = torch.unsqueeze(X, -1)
    B = (torch.matmul(X, torch.swapaxes(X, 1, -1)) * torch.conj(t))
    return B
