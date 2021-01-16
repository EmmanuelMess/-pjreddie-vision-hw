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

Using the layers proposed by [Karpathy](https://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html):
* Conv 5x5
* Max pool 2x2 stride 2
* Conv 5x5
* Max pool 2x2 stride 2
* Conv 5x5
* Max pool 2x2 stride 2
* linear 4*4*20x10 (RELU)

This seems plastic enough to learn to classify better than LazyNet and BoringNet.
But it does take around 4hs to train 200 epochs on Google Colab on CPU.

## 50 epochs default hiperparams
<img src="img/coolnet/basic/acc.png" />

Training seemed to be overfitting, although the test accuracy is much better than before at 70%.

## Learning rates

### 50 epochs learning rate 10

This broke completely, the network went to nan values very fast.

### 50 epochs learning rate 0.1

<img src="img/coolnet/lr 0.1/acc.png" />

Test accuracy goes down when compared to the default lr of 0.01. 
This might be because of 'skipping' over a region of lower loss, as the loss gets stuck around 1.2:

<img src="img/coolnet/lr 0.1/loss.png" />

Compared to default run with lr 0.01:

<img src="img/coolnet/basic/loss.png" />

### 50 epochs learning rate 0.01

This is the default, see "50 epochs default hiperparams"

### 50 epochs learning rate 0.0001

Very steady progress, although very slow:

<img src="img/coolnet/lr 0.0001/acc.png" />

It never even reaches 70% test accuracy.

### 0.01 seems best

The learning rate 0.01 seems to reach the 70% test accuracy mark fast enough, while not skipping over big areas of low loss:

<img src="img/coolnet/basic/loss.png" />

Loss reaches ~0.6 which seems ok.

## Reducing learning rate

Starting at 0.01 and slowly reducing every 50 epochs seems like a sensible idea, but it seems to worsen the test accuracy:

<img src="img/coolnet/reducing lr/acc.png" />

This might indicate that the network plasticity has reached its limit; that all the weights are stuck in a local minima; or that the training data is too small.

The third seems to be more accurate, see "Data Augmentation" lower down.

## Data Augmentation

Adding random transformation allow for us to be able to create more images from the existing 60k.

The transformations used are as follows:
* Random horizontal flipping (x2)
* Random affine transformation with rotation (>x2), horizontal translation (x4), vertical translation (x4), scaling (>x3)
* Random normalization: standard deviation can be 0.5, 0.7 or 0.9 (x3)

The last one was added to change the color of all pixels but still have the same image.

Here are a few transformed images:
<img src="img/coolnet/augmentation/example.png" />


### Results

Training for 200 epochs with reducing lr (10% less every 50 epochs):

<img src="img/coolnet/augmentation/acc.png" />

The resolution of the data is 50 epochs = one data point to accelerate the training.

The loss keeps going down during the training:

<img src="img/coolnet/augmentation/loss.png" />

It is clear the network is overfitting again.

### Changing the last layer of the network to softmax

Instead of RELU using sofmax allows the network to fail miserably:

<img src="img/coolnet/softmax/acc.png" />

### Negative Log Likelihood and log softmax

Changing the last layer to log softmax and using NLL seems to improve the test accuracy to >70% after 33 epochs. 