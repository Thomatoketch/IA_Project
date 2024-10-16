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
2) The optimizer plays a vital role in neural network training by minimizing the loss function, which measures the network's performance. 
It impacts the learning process in several key ways: determining the convergence speed, influencing generalization on unseen data, providing stability during training, and requiring hyperparameter tuning (like learning rates) for optimal results.
Additionally, some optimizers, such as Adam, offer better computational efficiency than others. Choosing the right optimizer and tuning it appropriately is crucial for achieving the best performance from the neural network.
3) Gradient Descent
Stochastic Gradient Descent
Stochastic Gradient Descent with Momentum
Mini Batch Gradient Descent
Adagrad (Adaptive Gradient Descent)
RMS Prop (Root Mean Square)
AdaDelta
Adam Optimizer
4) Gradient Descent : updates weights by computing the gradient over the entire dataset, which is efficient but slow for large datasets.
Stochastic Gradient Descent (SGD) : updates weights using individual samples, making it faster but less efficient due to noisy updates.
SGD with Momentum : improves efficiency and speed by adding momentum to dampen oscillations and stabilize learning.
Mini-Batch Gradient Descent : strikes a balance by updating weights using small batches, offering more stability than SGD and faster convergence than traditional Gradient Descent.
Adagrad : adapts learning rates for each parameter, making it more efficient for sparse data, though its learning rate decreases over time, slowing down learning.
RMSProp : improves upon Adagrad by maintaining a more constant learning rate, enhancing efficiency for non-convex problems.
AdaDelta : refines Adagrad further by overcoming the issue of decreasing learning rates, improving efficiency and speed.
Adam : combines the advantages of Momentum and RMSProp, offering the best balance of efficiency, speed, and stable convergence, making it one of the fastest and most widely used optimizers.
5) With the VGG architecture, the Adam optimizer delivers the best performance, achieving the highest validation accuracy in a reasonable training time. 
RMSprop also yields good accuracy but takes significantly longer to train compared to Adam. 
Stochastic Gradient Descent (SGD) trains the fastest but requires more iterations to reach Adam's accuracy. 
SGD with momentum offers no clear advantage over regular SGD and takes longer to train.

In summary, Adam is the most efficient and effective optimizer for the VGG architecture, while RMSprop performs well but is slower, and SGD is faster but less accurate without additional iterations.