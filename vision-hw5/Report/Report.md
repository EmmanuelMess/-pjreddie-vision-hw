# LazyNet
Fully connected one layer accuracy never goes above 40%:
<img src="img/lazynet/acc.png" />

This network seems to have too little plasticity to learn correctly.

# BoringNet
## non RELU
Without a non linear activation it seems that it can't correctly approximate at all.
<img src="img/boringnet/non relu/acc.png" />

## RELU
It looks like the network rapidly reaches ~50% accuracy in testing and then starts overfitting:
<img src="img/boringnet/relu/acc.png" />

The corresponding loss graph shows that, although the test accuracy doesn't increase, the (over) fitting to train data does:
<img src="img/boringnet/relu/loss.png" />

The RELU activation should allow the network to start being able to aproximate the function as described by the Universal Approximation Theorem.
But the amount of plasticity seems to still be too low.

# CoolNet

