# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    # warmup_epochs = args.epochs_base * args.warmup_rate
    warmup_epochs = args.epochs_base * args.warmup_rate / (args.warmup_rate + 1)
    if epoch < warmup_epochs:
        lr = args.lr_base * epoch / warmup_epochs 
    else:
        lr = args.min_lr + (args.lr_base - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - warmup_epochs) / (args.epochs_base - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    if lr < args.min_lr:
        lr = args.min_lr
    return lr