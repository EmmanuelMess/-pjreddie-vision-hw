from uwimg import *

def softmax_model(inputs, outputs):
    l = [make_layer(inputs, outputs, SOFTMAX)]
    return make_model(l)

def neural_net(inputs, outputs):
    print(inputs)
    l = [   make_layer(inputs, 32, RELU),
            make_layer(32, outputs, SOFTMAX)]
    return make_model(l)

def neural_net2(inputs, outputs):
    print(inputs)
    l = [   make_layer(inputs, 64, LRELU),
            make_layer(64, 32, LRELU),
            make_layer(32, outputs, SOFTMAX)]
    return make_model(l)


if __name__ == '__main__':
    print("loading data...")
    train = load_classification_data(b"cifar.train", b"cifar/labels.txt", 1)
    test  = load_classification_data(b"cifar.test", b"cifar/labels.txt", 1)
    print("done")
    print()

    print("training model...")
    batch = 128
    iters = 10000
    rate = 0.001
    momentum = .9
    decay = 0.01

    m = neural_net2(train.X.cols, train.y.cols)
    train_model(m, train, batch, iters, rate, momentum, decay)
    print("done")
    print()

    print("evaluating model...")
    print("training accuracy: {}".format(accuracy_model(m, train)))
    print("test accuracy:     {}".format(accuracy_model(m, test)))


## Questions ##

# 2.2.1 Why might we be interested in both training accuracy and testing accuracy? What do these two numbers tell us about our current model?

# Training accuracy allows the coder to know the level of underfitting or overfitting to training data.
# Testing accuracy allows the coder to know the percentage of correct classifications made by the model


# 2.2.2 Try varying the model parameter for learning rate to different powers of 10 (i.e. 10^1, 10^0, 10^-1, 10^-2, 10^-3) and training the model.
# What patterns do you see and how does the choice of learning rate affect both the loss during training and the final model accuracy?

# LR    | Final Loss | Acc Training | Acc Test
# 10^1  | -nan       | 0.0987       | 0.098
# 10^0  | 0.458577   | 0.8929       | 0.8889
# 10^-1 | 0.264909   | 0.9179       | 0.916
# 10^-2 | 0.330491   | 0.9024       | 0.9077
# 10^-3 | 0.596647   | 0.8586       | 0.869


# 2.2.3 Try varying the parameter for weight decay to different powers of 10: (10^0, 10^-1, 10^-2, 10^-3, 10^-4, 10^-5).
# How does weight decay affect the final model training and test accuracy?

# (LR 0.01)
# WD    | Final Loss | Acc Training | Acc Test
# 10^0  | 0.378454   | 0.8966       | 0.904
# 10^-1 | 0.334289   | 0.9018       | 0.9074
# 10^-2 | 0.330852   | 0.9023       | 0.9077
# 10^-3 | 0.330527   | 0.9024       | 0.9077
# 10^-4 | 0.330494   | 0.9024       | 0.9077
# 10^-5 | 0.330491   | 0.9024       | 0.9077


# 2.3.1 Currently the model uses a logistic activation for the first layer. Try using a the different activation functions we programmed.
# How well do they perform? What's best?

# (LR 0.01 WD 0)
# Activation | Final Loss | Acc Training | Acc Test | Best
# LINEAR     | 0.139125   | 0.9123       | 0.9134   | No
# LOGISTIC   | 0.433090   | 0.8580       | 0.863    | No
# RELU       | 0.132344   | 0.9231       | 0.924    | Yes
# LRELU      | 0.133097   | 0.9207       | 0.9217   | No
# SOFTMAX    | 1.135004   | 0.5814       | 0.5867   | No (this NN doesn't make sense)


# 2.3.2 Using the same activation, find the best (power of 10) learning rate for your model.
# What is the training accuracy and testing accuracy?

# (WD 0 Activation RELU)
# LR    | Final Loss       | Acc Training | Acc Test | Best
# 10^1  | -nan             | 0.0987       | 0.098    | No
# 10^0  | 1.782362 or -nan | 0.2080       | 0.2093   | No
# 10^-1 | 0.070356         | 0.9612       | 0.9564   | Yes
# 10^-2 | 0.132344         | 0.9231       | 0.924    | No
# 10^-3 | 0.410790         | 0.8644       | 0.8682   | No


# 2.3.3 Right now the regularization parameter `decay` is set to 0. Try adding some decay to your model.
# What happens, does it help? Why or why not may this be?

# (LR 10^-1 Activation RELU)
# WD    | Final Loss | Acc Training | Acc Test | Best
# 10^0  | 0.178535   | 0.9245       | 0.9272   | No
# 10^-1 | 0.076561   | 0.9569       | 0.952    | No
# 10^-2 | 0.083339   | 0.9620       | 0.9563   | No
# 10^-3 | 0.092674   | 0.9619       | 0.9566   | No
# 10^-4 | 0.064849   | 0.9625       | 0.9586   | Yes
# 10^-5 | 0.069342   | 0.9606       | 0.956    | No


# 2.3.4 Modify your model so it has 3 layers instead of two. The layers should be `inputs -> 64`, `64 -> 32`, and `32 -> outputs`.
# Also modify your model to train for 3000 iterations instead of 1000.
# Look at the training and testing error for different values of decay (powers of 10, 10^-4 -> 10^0).
# Which is best? Why?

# (LR 10^-1 Activation RELU)
# WD    | Final Loss | Acc Training | Acc Test | Best
# 10^0  | 0.181912   | 0.9419       | 0.9442   | No
# 10^-1 | 0.058319   | 0.9779       | 0.9725   | No
# 10^-2 | 0.026900   | 0.9822       | 0.9706   | No
# 10^-3 | 0.057030   | 0.9833       | 0.9712   | No
# 10^-4 | 0.030254   | 0.9870       | 0.9751   | Yes


# 3.2.1 How well does your network perform on the CIFAR dataset?

# (Ordered by ascending test accuracy)
# Iters | Batch size | WD    | LR    | Activation 1 | Activation 2 | Final Loss | Acc Training | Acc Test
#  5k   | 128        | 10^-5 | 10^-6 | LRELU        | LRELU        | 2.312677   | 0.1121       | 0.1096
#  3k   | 128        | 10^-4 | 10^-1 | RELU         | RELU         | 2.162171   | 0.1236       | 0.1215
#  5k   | 128        | 0     | 10^-3 | LOGISTIC     | LRELU        | 1.926278   | 0.3070       | 0.2991
#  5k   | 128        | 0     | 10^-4 | LRELU        | LRELU        | 1.923621   | 0.3183       | 0.3161
#  5k   | 128        | 10^-3 | 10^-4 | LRELU        | LRELU        | 1.923650   | 0.3182       | 0.3165
#  1k   | 128        | 10^-4 | 10^-3 | RELU         | RELU         | 1.807361   | 0.3376       | 0.3323
#  1k   | 128        | 0     | 10^-3 | RELU         | RELU         | 1.807224   | 0.3370       | 0.3325
#  1k   | 128        | 0     | 10^-3 | LRELU        | LRELU        | 1.805216   | 0.3415       | 0.3371
#  2.5k | 512        | 10^-5 | 10^-3 | LRELU        | LRELU        | 1.680667   | 0.3956       | 0.3982
#  5k   | 128        | 10^-5 | 10^-3 | LRELU        | LRELU        | 1.531603   | 0.4346       | 0.4279
#  5k   | 128        | 0     | 10^-3 | LRELU        | LRELU        | 1.530942   | 0.4338       | 0.4289
# 10k   | 128        | 10^-2 | 10^-3 | LRELU        | LRELU        | 1.444736   | 0.4792       | 0.4580
# 10k   | 128        | 0     | 10^-3 | LRELU        | LRELU        | 1.448380   | 0.4798       | 0.4596
# 10k   | 128        | 10^-5 | 10^-3 | LRELU        | LRELU        | 1.450659   | 0.4803       | 0.4605
