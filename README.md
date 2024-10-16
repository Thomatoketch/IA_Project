projet IA

Task 1 : 
3) number of training samples : 50000 
number of test samples : 10000
4) shape of each image : 32x32x3 (taille de l'image L*l*profondeur(3 pour les images couleurs RGB))
channel color : 3 RGB

Task 2 :
2) The label of data represents its name.
the label is numerical.
3) very blur for humans
it depends of the label but a major part is easy to recognize for humans

Task 3 :
2) It help stabalize the data and make it easier to use for the model and converge faster to the optimal solution. 
It could happen that the model don't convege to the optimal solution at all.
3) As the images are 32 x 32 x 3 in size, normalization makes it possible to avoid very large values.
4) faster training, highter accuracy and reduce sensitivity to initial weights

Task 4 :
1) fonction prise de ce site : https://stackoverflow.com/questions/59062582/one-hot-encoding-from-image-labels-using-numpy
2) On ne peut pas utiliser les labels directement car ils sont sous forme de string et non de nombre. 
3) One-hot encoding allow us to compare y_pred and y_true because we can't compare strings.

Task 5 :
2) There is the same amount of each class in ther CIFAR-10 dataset.
3) It's important to have the same amount of each class in the training set to avoid bias in the model.



Part 2 :
1) Optimizers in deep learning are algorithms that adjust model parameters during training to minimize the loss function, significantly influencing the speed and quality of convergence.
Common optimizers include SGD, Adam, RMSprop, and Adagrad, each with distinct strategies for updating parameters and handling learning rates.