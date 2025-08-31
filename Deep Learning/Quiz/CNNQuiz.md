# CNN Quiz

**1. A convolutional layer has an input of size 64x64x16, and uses 32 filters of size 3x3 with a stride of 1 and padding of 1. What is the total number of learnable parameters in this layer?**
- [ ] A) 147456
- [ ] B) 4640
- [ ] C) 4608
- [ ] D) 147488

**Correct Answer:** B

**Explanation:**
- A) Incorrect. This is a miscalculation.
- B) Correct. The number of weights is (filter_height * filter_width * input_channels) * num_filters = (3 * 3 * 16) * 32 = 4608. The number of biases is equal to the number of filters, which is 32. Total parameters = 4608 + 32 = 4640.
- C) Incorrect. This is only the number of weights, without considering the biases.
- D) Incorrect. This calculation is likely a result of multiplying the output dimensions with the parameters.

**2. What is the primary motivation for using Global Average Pooling (GAP) instead of a traditional flattening operation before the final dense layers in a CNN?**
- [ ] A) To enforce a 1-to-1 correspondence between feature maps and class labels, while drastically reducing the number of parameters and preventing overfitting.
- [ ] B) To introduce more non-linearity into the network.
- [ ] C) To preserve the spatial dimensions of the feature maps.
- [ ] D) To increase the number of parameters and model complexity.

**Correct Answer:** A

**Explanation:**
- A) Correct. GAP averages out the spatial information in each feature map, resulting in one value per map. This creates a much smaller feature vector than flattening, which reduces overfitting. It also encourages the network to create feature maps that are more directly correlated with the final classes.
- B) Incorrect. GAP is a linear operation (averaging).
- C) Incorrect. The purpose of GAP is to collapse the spatial dimensions.
- D) Incorrect. GAP significantly reduces the number of parameters.

**3. Which optimization algorithm combines the benefits of both Momentum and RMSProp?**
- [ ] A) L-BFGS
- [ ] B) Adagrad
- [ ] C) Adam
- [ ] D) Stochastic Gradient Descent (SGD)

**Correct Answer:** C

**Explanation:**
- A) Incorrect. L-BFGS is a quasi-Newton method, which is a different class of optimizer not typically used for deep neural networks.
- B) Incorrect. Adagrad is an adaptive learning rate algorithm, but it's not the one that combines Momentum and RMSProp.
- C) Correct. Adam (Adaptive Moment Estimation) uses both a moving average of the past gradients (like Momentum) and a moving average of the squared gradients (like RMSProp) to adapt the learning rate for each parameter.
- D) Incorrect. SGD is a basic optimizer that both Momentum and RMSProp improve upon.

**4. A Cosine learning rate scheduler is often preferred over a simple Step scheduler because:**
- [ ] A) It only works with the Adam optimizer.
- [ ] B) It provides a smooth, gradual decay of the learning rate, which can help the model settle into a better minimum without the abrupt changes of a step decay.
- [ ] C) It increases the learning rate linearly over time.
- [ ] D) It keeps the learning rate constant for the entire training process.

**Correct Answer:** B

**Explanation:**
- A) Incorrect. Learning rate schedulers are generally optimizer-agnostic.
- B) Correct. The cosine annealing schedule smoothly decreases the learning rate, which can be more effective than the sudden drops of a step scheduler, often leading to better convergence.
- C) Incorrect. This describes a linear scheduler.
- D) Incorrect. This describes a constant learning rate.

**5. In a multi-class classification problem with 10 classes, which loss function would be the most appropriate choice?**
- [ ] A) Hinge Loss
- [ ] B) Mean Squared Error (MSE)
- [ ] C) Binary Cross-Entropy (BCE)
- [ ] D) Categorical Cross-Entropy (CCE)

**Correct Answer:** D

**Explanation:**
- A) Incorrect. Hinge loss is typically associated with Support Vector Machines (SVMs).
- B) Incorrect. MSE is used for regression tasks, where the output is a continuous value.
- C) Incorrect. BCE is used for binary (two-class) classification problems.
- D) Correct. CCE is specifically designed for multi-class classification problems where each sample belongs to exactly one class.

**6. What is the output size of a convolutional layer with an input of 32x32, a filter size of 3x3, a stride of 2, and padding of 1?**
- [ ] A) 17x17
- [ ] B) 15x15
- [ ] C) 32x32
- [ ] D) 16x16

**Correct Answer:** D

**Explanation:**
- A) Incorrect. This is a miscalculation.
- B) Incorrect. This would be the result of not adding 1 at the end of the formula.
- C) Incorrect. The output size would only be the same if the stride was 1.
- D) Correct. Using the formula `o = (I - k + 2P) / S + 1`, we get `(32 - 3 + 2*1) / 2 + 1 = 31 / 2 + 1 = 15.5 + 1 = 16.5`. Since we round down, the output is 16x16.

**7. Which of the following deep learning architectures is most suited for processing sequential data like text or time series?**
- [ ] A) Long Short-Term Memory (LSTM)
- [ ] B) Generative Adversarial Network (GAN)
- ] C) Autoencoder (AE)
- [ ] D) Convolutional Neural Network (CNN)

**Correct Answer:** A

**Explanation:**
- A) Correct. LSTMs are a type of recurrent neural network (RNN) that are highly effective at learning from sequential data due to their internal memory cells.
- B) Incorrect. GANs are used for generating new data.
- C) Incorrect. Autoencoders are primarily used for unsupervised learning and dimensionality reduction.
- D) Incorrect. While CNNs can be used for sequence data (1D convolutions), LSTMs are specifically designed for this purpose.

**8. The concept of "parameter sharing" in a convolutional layer refers to:**
- [ ] A) Using the same pooling operation for all feature maps.
- [ ] B) Using the same filter (weights and bias) to scan across all spatial locations of the input.
- [ ] C) Sharing the same learning rate for all parameters in the layer.
- [ ] D) Using the same activation function for all neurons in the layer.

**Correct Answer:** B

**Explanation:**
- A) Incorrect. This is not related to parameter sharing in the convolutional layer itself.
- B) Correct. This is the key idea behind convolutions. The same filter is applied to different parts of the image, which drastically reduces the number of parameters compared to a fully connected layer and allows the network to detect features regardless of their location.
- C) Incorrect. While a base learning rate might be shared, adaptive optimizers will change it for each parameter.
- D) Incorrect. This is standard practice but not what parameter sharing refers to.

**9. What is the primary role of the Softmax activation function in a CNN classifier?**
- [ ] A) To reduce the number of parameters.
- [ ] B) To introduce non-linearity.
- [ ] C) To convert the raw output scores (logits) of the final dense layer into a probability distribution over the classes.
- [ ] D) To normalize the feature maps.

**Correct Answer:** C

**Explanation:**
- A) Incorrect. Softmax does not affect the number of parameters.
- B) Incorrect. While Softmax is non-linear, its primary purpose is not just to introduce non-linearity like ReLU.
- C) Correct. Softmax takes a vector of arbitrary real-valued scores and squashes it into a vector of values between 0 and 1 that sum to 1, which can be interpreted as class probabilities.
- D) Incorrect. This is the role of Batch Normalization.

**10. Data augmentation techniques like rotating, flipping, and zooming are used to:**
- [ ] A) Normalize the input data.
- [ ] B) Reduce the number of learnable parameters.
- [ ] C) Artificially increase the size and diversity of the training data to help the model generalize better and prevent overfitting.
- [ ] D) Speed up the training process.

**Correct Answer:** C

**Explanation:**
- A) Incorrect. Normalization is a separate preprocessing step.
- B) Incorrect. It does not affect the number of parameters in the model.
- C) Correct. By creating modified versions of the training images, data augmentation exposes the model to a wider variety of data, which helps it learn more robust features and perform better on unseen data.
- D) Incorrect. Data augmentation can sometimes slow down training slightly due to the extra processing, but this is not its purpose.

**11. What problem does the RMSProp optimizer primarily address?**
- [ ] A) The aggressive, monotonically decreasing learning rate of Adagrad.
- [ ] B) The slow convergence of SGD by using momentum.
- [ ] C) The inability of neural networks to learn non-linear functions.
- [ ] D) The vanishing gradient problem.

**Correct Answer:** A

**Explanation:**
- A) Correct. RMSProp is an adaptive learning rate method that, unlike Adagrad, uses a moving average of squared gradients, which prevents the learning rate from becoming too small too quickly.
- B) Incorrect. This is addressed by Momentum SGD.
- C) Incorrect. This is solved by using non-linear activation functions.
- D) Incorrect. While adaptive optimizers can help with vanishing gradients, this is not their primary purpose.

**12. A Transformer architecture is most commonly associated with which type of task?**
- [ ] A) Unsupervised clustering.
- [ ] B) Time series forecasting.
- [ ] C) Natural Language Processing (NLP) and Large Language Models (LLMs).
- [ ] D) Image classification.

**Correct Answer:** C

**Explanation:**
- A) Incorrect. This is the domain of algorithms like K-Means or Autoencoders.
- B) Incorrect. While Transformers can be used for time series, LSTMs are also very common.
- C) Correct. The Transformer architecture, with its self-attention mechanism, revolutionized NLP and is the foundation for models like GPT and Gemini.
- D) Incorrect. While Vision Transformers (ViTs) exist, the Transformer architecture was originally developed for and is most dominant in NLP.

**13. In the context of a convolutional layer, what does the "stride" hyperparameter control?**
- [ ] A) The number of pixels the filter jumps as it moves across the input.
- [ ] B) The size of the filter.
- [ ] C) The amount of zero-padding added to the input.
- [ ] D) The number of filters in the layer.

**Correct Answer:** A

**Explanation:**
- A) Correct. A stride of 1 means the filter moves one pixel at a time. A stride of 2 means it jumps two pixels, resulting in a smaller output feature map.
- B) Incorrect. This is the `kernel_size`.
- C) Incorrect. This is the `padding`.
- D) Incorrect. This is controlled by the `filters` or `out_channels` hyperparameter.

**14. Which of the following is a key characteristic of a Generative Adversarial Network (GAN)?**
- [ ] A) It is primarily used for dimensionality reduction.
- [ ] B) It consists of two competing neural networks, a generator and a discriminator, which are trained simultaneously.
- [ ] C) It is a supervised learning technique for regression.
- [ ] D) It uses a single network to classify images.

**Correct Answer:** B

**Explanation:**
- A) Incorrect. This is the primary use of Autoencoders.
- B) Correct. The generator tries to create realistic data, while the discriminator tries to distinguish between real and fake data. This adversarial process leads to the generation of high-quality synthetic data.
- C) Incorrect. GANs are a form of unsupervised or semi-supervised learning for generation.
- D) Incorrect. A GAN has two networks.

**15. What is the purpose of the "padding" hyperparameter in a convolutional layer?**
- [ ] A) To increase the stride of the convolution.
- [ ] B) To increase the non-linearity of the layer.
- [ ] C) To control the spatial size of the output feature map and to allow the filter to properly process the edges of the input.
- [ ] D) To reduce the number of parameters.

**Correct Answer:** C

**Explanation:**
- A) Incorrect. Stride is a separate hyperparameter.
- B) Incorrect. Padding is a linear operation.
- C) Correct. By adding a border of zeros around the input, padding allows the filter to be centered on the edge pixels. "Same" padding is often used to ensure the output feature map has the same spatial dimensions as the input.
- D) Incorrect. Padding does not affect the number of parameters.

**16. If a convolutional layer has an input of 28x28x3 and uses 16 filters of size 3x3 with a stride of 1 and no padding, what is the depth of the output volume?**
- [ ] A) 1
- [ ] B) 16
- [ ] C) 26
- [ ] D) 3

**Correct Answer:** B

**Explanation:**
- A) Incorrect. This would only be the case if there was one filter.
- B) Correct. The depth of the output volume of a convolutional layer is always equal to the number of filters used in that layer.
- C) Incorrect. 26 is the spatial dimension (width and height) of the output, not the depth.
- D) Incorrect. 3 is the depth of the input volume.

**17. Which of the following statements about the backpropagation algorithm is true?**
- [ ] A) It is a method for data augmentation.
- [ ] B) It is an algorithm for calculating the gradient of the loss function with respect to the network's weights, which is then used by an optimizer to update the weights.
- [ ] C) It is only used in the final layer of the network.
- ] D) It is used to calculate the loss of the network during the forward pass.

**Correct Answer:** B

**Explanation:**
- A) Incorrect. It is a training algorithm, not a data preprocessing technique.
- B) Correct. Backpropagation is the core algorithm that allows neural networks to learn. It efficiently computes the gradients by propagating the error backward through the network, from the output layer to the input layer.
- C) Incorrect. It is used to calculate gradients for all layers.
- D) Incorrect. The loss is calculated after the forward pass.

**18. An Autoencoder (AE) is an unsupervised learning model that is primarily used for:**
- [ ] A) Learning a compressed representation (encoding) of the input data, often for dimensionality reduction or feature learning.
- [ ] B) Generating new images from random noise.
- [ ] C) Predicting future values in a time series.
- ] D) Classifying data into multiple categories.

**Correct Answer:** A

**Explanation:**
- A) Correct. An autoencoder consists of an encoder that compresses the input into a low-dimensional latent space, and a decoder that reconstructs the input from this representation. This forces the network to learn the most important features of the data.
- B) Incorrect. This is the primary use of GANs or Diffusion Models.
- C) Incorrect. This is a time series forecasting task, typically handled by LSTMs or other sequence models.
- D) Incorrect. This is a supervised classification task.

**19. What is a significant drawback of using a very small, fixed learning rate during the entire training process?**
- [ ] A) It will cause the gradients to vanish.
- [ ] B) The training process will be very slow, and the model may get stuck in a poor local minimum.
- [ ] C) The model will overfit to the training data.
- ] D) The model will likely diverge and the loss will explode.

**Correct Answer:** B

**Explanation:**
- A) Incorrect. The vanishing gradient problem is related to the network architecture and activation functions, not the learning rate itself.
- B) Correct. A learning rate that is too small will cause the model to make very small steps in the loss landscape, leading to slow convergence. It may also lack the "momentum" to escape shallow local minima.
- C) Incorrect. Overfitting is not a direct consequence of a small learning rate.
- D) Incorrect. A small learning rate is unlikely to cause divergence; a large one would.

**20. In a CNN, the term "feature map" refers to:**
- [ ] A) The set of all learnable parameters in a layer.
- [ ] B) The output of a filter applied to the input, which highlights the locations of a specific feature.
- [ ] C) The final classification output.
- ] D) The input image.

**Correct Answer:** B

**Explanation:**
- A) Incorrect. These are the weights and biases.
- B) Correct. Each filter in a convolutional layer is trained to detect a specific feature (like a vertical edge). The feature map is the 2D grid of activations that results from sliding that filter over the input, showing where that feature is present.
- C) Incorrect. This is the output of the final classifier.
- D) Incorrect. The input image is the starting point.

**21. Which of the following is NOT a hyperparameter of a convolutional layer?**
- [ ] A) Padding
- [ ] B) Stride
- [ ] C) The weights of the filters
- ] D) Filter size

**Correct Answer:** C

**Explanation:**
- A) Incorrect. Padding is a hyperparameter that must be set before training.
- B) Incorrect. Stride is a hyperparameter that must be set before training.
- C) Correct. The weights of the filters are the parameters that the network learns during the training process. They are not set by the user beforehand.
- D) Incorrect. Filter size (or kernel size) is a hyperparameter that must be set before training.

**22. The main purpose of a validation set in the training process is to:**
- [ ] A) Augment the training data.
- [ ] B) Provide a final, unbiased evaluation of the model's performance after training is complete.
- [ ] C) Tune the model's hyperparameters and monitor for overfitting during training.
- ] D) Train the model.

**Correct Answer:** C

**Explanation:**
- A) Incorrect. The validation set should be kept separate from the training data.
- B) Incorrect. This is the purpose of the test set, which the model never sees during training or hyperparameter tuning.
- C) Correct. The validation set is used to evaluate the model's performance on unseen data during training. This helps in tuning hyperparameters (like the learning rate) and in identifying the point at which the model starts to overfit (i.e., when the training loss continues to decrease but the validation loss starts to increase).
- D) Incorrect. The model is trained on the training set.

**23. A Linear learning rate scheduler:**
- [ ] A) Follows a cosine curve.
- [ ] B) Decreases the learning rate by a fixed factor every few epochs.
- [ ] C) Decreases the learning rate linearly from an initial value to a final value over the course of training.
- ] D) Keeps the learning rate constant.

**Correct Answer:** C

**Explanation:**
- A) Incorrect. This is a cosine scheduler.
- B) Incorrect. This is a step scheduler.
- C) Correct. A linear scheduler provides a smooth, steady decrease in the learning rate throughout the training process.
- D) Incorrect. This is a constant scheduler.

**24. What is the effect of using a stride of 2 instead of 1 in a convolutional layer?**
- [ ] A) It has no effect on the output.
- [ ] B) It reduces the spatial dimensions of the output feature map.
- [ ] C) It increases the number of learnable parameters.
- ] D) It increases the spatial dimensions of the output feature map.

**Correct Answer:** B

**Explanation:**
- A) Incorrect. The stride has a significant effect on the output size.
- B) Correct. A stride of 2 means the filter jumps 2 pixels at a time, effectively downsampling the input and producing a smaller feature map. This is often used to reduce the computational cost.
- C) Incorrect. The stride does not affect the number of parameters in the filters.
- D) Incorrect. A larger stride results in a smaller output.

**25. Which of the following is a key difference between a Variational Autoencoder (VAE) and a standard Autoencoder (AE)?**
- [ ] A) VAEs do not have a decoder.
- [ ] B) VAEs encode the input into a probability distribution in the latent space, while AEs encode it into a single point.
- [ ] C) VAEs cannot be used for dimensionality reduction.
- ] D) VAEs are used for supervised learning, while AEs are unsupervised.

**Correct Answer:** B

**Explanation:**
- A) Incorrect. Both have an encoder and a decoder.
- B) Correct. This is the core difference. By learning a distribution (a mean and variance) for the latent space, VAEs can generate new data by sampling from this distribution. Standard AEs are not generative in the same way.
- C) Incorrect. VAEs can be used for dimensionality reduction, just like AEs.
- D) Incorrect. Both are typically used for unsupervised learning.

**26. The "hierarchical feature extraction" in a CNN means that:**
- [ ] A) The features are organized by their color.
- [ ] B) The network has a tree-like structure.
- [ ] C) Deeper layers of the network learn to recognize more complex and abstract features by combining the simpler features learned by the earlier layers.
- ] D) Features are extracted in a random order.

**Correct Answer:** C

**Explanation:**
- A) Incorrect. Color is just one of many simple features that might be learned by early layers.
- B) Incorrect. The structure is typically sequential.
- C) Correct. This is the fundamental principle of how CNNs learn to see. Early layers might find edges, the next might find textures, the next might find object parts, and so on.
- D) Incorrect. There is a clear progression from simple to complex.

**27. If your task is to predict the price of a house based on its features, which loss function would be most appropriate?**
- [ ] A) Cosine Similarity
- [ ] B) Categorical Cross-Entropy (CCE)
- [ ] C) Mean Squared Error (MSE)
- ] D) Binary Cross-Entropy (BCE)

**Correct Answer:** C

**Explanation:**
- A) Incorrect. Cosine similarity measures the angle between two vectors, not the difference between two continuous values.
- B) Incorrect. CCE is for multi-class classification.
- C) Correct. Predicting a house price is a regression task, as the output is a continuous value. MSE measures the average squared difference between the predicted and actual prices, which is a standard loss function for regression.
- D) Incorrect. BCE is for binary classification.

**28. What is the main advantage of using a Max Pooling layer?**
- [ ] A) It normalizes the activations.
- [ ] B) It introduces non-linearity.
- [ ] C) It reduces the spatial dimensions of the feature maps, which reduces computation and provides a degree of translational invariance.
- ] D) It increases the number of feature maps.

**Correct Answer:** C

**Explanation:**
- A) Incorrect. This is the role of Batch Normalization.
- B) Incorrect. This is the role of activation functions.
- C) Correct. Max pooling downsamples the feature maps, making the network more computationally efficient. It also makes the network more robust to the exact location of features in the input.
- D) Incorrect. The number of feature maps remains the same.

**29. The Adam optimizer is considered "adaptive" because it:**
- [ ] A) Adapts the activation functions during training.
- [ ] B) Adapts the learning rate for each parameter individually based on past gradients.
- [ ] C) Adapts the batch size during training.
- ] D) Adapts the network architecture during training.

**Correct Answer:** B

**Explanation:**
- A) Incorrect. Activation functions are fixed.
- B) Correct. Adam maintains a separate learning rate for each weight and adapts it based on the first and second moments of the gradients, which allows for faster convergence.
- C) Incorrect. The batch size is a hyperparameter that is usually fixed.
- D) Incorrect. The architecture is fixed before training.

**30. Which of the following is NOT a common data augmentation technique for images?**
- [ ] A) Zooming
- [ ] B) Flipping
- [ ] C) Backpropagation
- ] D) Rotation

**Correct Answer:** C

**Explanation:**
- A) Incorrect. Zooming in or out is a common augmentation technique.
- B) Incorrect. Horizontal flipping is a very common augmentation technique.
- C) Correct. Backpropagation is the algorithm used to train the neural network by updating its weights; it is not a data preprocessing or augmentation technique.
- D) Incorrect. Rotation is a common augmentation technique.

### Back to Reading Content --> [Convolutional Neural Networks (CNNs)](../CNN.md)