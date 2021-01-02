#ifndef VISION_HW4_CLASSIFIER_H
#define VISION_HW4_CLASSIFIER_H

layer make_layer(int input, int output, ACTIVATION activation);
void train_model(model m, data d, int batch, int iters, double rate, double momentum, double decay);
double accuracy_model(model m, data d);

#endif //VISION_HW4_CLASSIFIER_H
