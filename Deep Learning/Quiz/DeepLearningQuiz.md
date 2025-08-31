
**1. In a Transformer's self-attention mechanism, if the Key and Value matrices were identical (K=V), what would be the primary implication?**
- [ ] A) The model would be unable to compute attention scores.
- [ ] B) The output of the attention layer would be a simple weighted average of the input embeddings themselves, potentially limiting its expressive power.
- [ ] C) The Query vectors would become redundant.
- [ ] D) The model would be more efficient as it requires fewer matrix multiplications.

**Correct Answer:** B

**Explanation:**
- A) Attention scores are calculated from Q and K, so this would still be possible.
- B) The attention weights determine how to average the Value vectors. If K=V, the model is averaging the same vectors that it used to determine the weights, which is less expressive than having separate Key and Value representations.
- C) The Query vectors are still needed to determine the attention weights.
- D) The number of matrix multiplications would be the same.

**2. A key reason LSTMs are more effective than simple RNNs at capturing long-range dependencies is that:**
- [ ] A) The cell state acts as a high-bandwidth channel for gradient flow, primarily using additive interactions that are less prone to vanishing.
- [ ] B) LSTMs use the ReLU activation function, which prevents gradients from shrinking.
- [ ] C) The backpropagation through time (BPTT) algorithm is fundamentally different for LSTMs.
- [ ] D) LSTMs have more learnable parameters than simple RNNs.

**Correct Answer:** A

**Explanation:**
- A) The cell state's additive nature and the gating mechanism provide a more direct path for gradients to flow through time, mitigating the vanishing gradient problem.
- B) LSTMs typically use sigmoid and tanh activation functions, not ReLU.
- C) BPTT is the standard training algorithm for both.
- D) While true, the number of parameters is not the reason for their effectiveness at this specific task.

**3. In a Variational Autoencoder (VAE), the reparameterization trick is essential because:**
- [ ] A) It ensures the latent space is continuous.
- [ ] B) It allows gradients to be backpropagated through a stochastic (random sampling) node.
- [ ] C) It reduces the dimensionality of the latent space.
- [ ] D) It makes the decoder more powerful.

**Correct Answer:** B

**Explanation:**
- A) The continuity of the latent space is a result of the KL divergence loss term, not the reparameterization trick itself.
- B) Backpropagation cannot flow through a random node. The trick reformulates the sampling process to make the randomness an input, allowing gradients to flow back to the learnable parameters of the latent distribution (mean and variance).
- C) The dimensionality is a design choice.
- D) It does not directly affect the decoder's architecture.

**4. The primary role of the cross-attention mechanism in a Transformer decoder is to:**
- [ ] A) Allow the decoder to attend to its own previously generated tokens.
- [ ] B) Integrate the positional encodings with the token embeddings.
- [ ] C) Allow the decoder to focus on relevant parts of the encoder's output (the source sequence).
- [ ] D) Combine the outputs of the multiple attention heads.

**Correct Answer:** C

**Explanation:**
- A) This is the role of the masked self-attention layer in the decoder.
- B) This happens at the input layer.
- C) This is the crucial step where the decoder uses the context from the input sentence to generate the next token in the output sentence.
- D) This is done within each multi-head attention block.

**5. A significant advantage of a Convolutional Autoencoder (CAE) over a fully-connected autoencoder for image data is that:**
- [ ] A) CAEs are a form of lossless compression.
- [ ] B) CAEs preserve the spatial hierarchy of the input image through the use of convolutional layers.
- [ ] C) CAEs require significantly fewer training examples.
- [ ] D) CAEs do not require a bottleneck layer.

**Correct Answer:** B

**Explanation:**
- A) All autoencoders are a form of lossy compression.
- B) By using convolutional filters, CAEs can learn and reconstruct spatial features, which is much more effective for images than a fully-connected architecture that flattens the input.
- C) The amount of training data required depends on the complexity of the task, not the architecture itself.
- D) The bottleneck is a defining feature of all autoencoders.

**6. In a Diffusion Model, the reverse process (denoising) is a learned process, while the forward process (noising) is:**
- [ ] A) Also a learned process, trained adversarially against the reverse process.
- [ ] B) A fixed, mathematical process where noise is added according to a predefined schedule.
- [ ] C) A process that uses a VAE to add structured noise.
- [ ] D) A process that is only defined after the reverse process has been fully trained.

**Correct Answer:** B

**Explanation:**
- B) The forward process is a simple, non-learned Markov chain that gradually adds Gaussian noise to the data. The model only learns how to reverse this fixed process.

**7. The "vanishing gradient" problem in RNNs is primarily caused by:**
- [ ] A) The use of shared weights across all time steps.
- [ ] B) The repeated multiplication of gradients through time, especially from activation functions whose derivatives are less than 1.
- [ ] C) The use of the Softmax function in the output layer.
- [ ] D) The sequential nature of the input data.

**Correct Answer:** B

**Explanation:**
- B) During backpropagation through time, the chain rule requires multiplying the gradients at each step. For activation functions like tanh, these gradients are often small, and their repeated multiplication causes the overall gradient to shrink exponentially.

**8. What is the fundamental difference between how a GAN and a VAE generate new data?**
- [ ] A) GANs are trained with a discriminator in an adversarial process, while VAEs are trained to reconstruct their input.
- [ ] B) VAEs can only generate blurry images, while GANs can only generate sharp images.
- [ ] C) GANs learn a continuous latent space, while VAEs learn a discrete one.
- [ ] D) VAEs are supervised models, while GANs are unsupervised.

**Correct Answer:** A

**Explanation:**
- A) This is the core architectural and training difference. GANs use a two-player game to learn the data distribution implicitly, while VAEs use a reconstruction loss and a regularization term (KL divergence) to explicitly model the data distribution.

**9. The purpose of the "mask" in the masked multi-head self-attention layer of a Transformer decoder is to:**
- [ ] A) Prevent the model from attending to padding tokens.
- [ ] B) Randomly drop out attention heads to prevent overfitting.
- [ ] C) Prevent the decoder from attending to future tokens in the sequence during training.
- [ ] D) Hide the query vectors from the key vectors.

**Correct Answer:** C

**Explanation:**
- C) Since the decoder generates the output autoregressively (one token at a time), it should not be allowed to "see" the future tokens in the target sequence during training. The mask enforces this by setting the attention scores for future positions to negative infinity.

**10. A key benefit of using Global Average Pooling (GAP) instead of a Flatten layer before a CNN's classifier is that it:**
- [ ] A) Increases the number of parameters, allowing for a more complex decision boundary.
- [ ] B) Drastically reduces the number of parameters, making the model less prone to overfitting.
- [ ] C) Preserves the spatial information of the feature maps.
- [ ] D) Can only be used with very deep networks.

**Correct Answer:** B

**Explanation:**
- B) A Flatten layer can create a huge number of parameters. GAP reduces each feature map to a single value, creating a much smaller feature vector and thus reducing the risk of overfitting in the final dense layers.

**11. The forget gate in an LSTM cell is responsible for:**
- [ ] A) Deciding which new information to add to the cell state.
- [ ] B) Determining which information from the previous cell state should be discarded.
- [ ] C) Filtering the cell state to produce the hidden state.
- [ ] D) Resetting the entire cell state to zero at each time step.

**Correct Answer:** B

**Explanation:**
- B) The forget gate looks at the current input and previous hidden state to decide which parts of the long-term memory (the cell state) are no longer relevant and should be forgotten.

**12. The sinusoidal positional encodings in the original Transformer are designed to:**
- [ ] A) Be learned during the training process.
- [ ] B) Allow the model to easily learn relative positions and generalize to sequences of unseen lengths.
- [ ] C) Convert the input tokens into a continuous representation.
- [ ] D) Reduce the dimensionality of the input embeddings.

**Correct Answer:** B

**Explanation:**
- B) The properties of sine and cosine functions allow the model to represent positions in a way that makes it easy to understand relative positioning (e.g., the relationship between position `pos` and `pos + k` is a linear transformation), and they can be extrapolated to any sequence length.

**13. In a GAN, if the discriminator's loss converges to zero while the generator's loss explodes, this is a sign of:**
- [ ] A) Successful training and convergence.
- [ ] B) The generator successfully fooling the discriminator.
- [ ] C) Mode collapse, where the generator produces a limited variety of samples.
- [ ] D) The discriminator becoming too powerful and the generator failing to learn.

**Correct Answer:** D

**Explanation:**
- D) If the discriminator's loss is zero, it means it can perfectly distinguish real from fake images. When this happens, the generator receives no useful gradient signal from the discriminator and cannot improve, leading to training failure.

**14. The primary advantage of a Convolutional Neural Network (CNN) over a standard Artificial Neural Network (ANN) for image classification is:**
- [ ] A) CNNs require less training data.
- [ ] B) CNNs use parameter sharing and local connectivity to preserve spatial hierarchies and reduce the number of parameters.
- [ ] C) CNNs are faster to train because they have fewer layers.
- [ ] D) ANNs cannot be used for classification tasks.

**Correct Answer:** B

**Explanation:**
- B) CNNs are designed to respect the spatial structure of images. Parameter sharing (using the same filter across the image) and local connectivity (filters looking at small patches) make them highly efficient and effective at learning visual features.

**15. The loss function of a VAE includes a reconstruction term and a KL divergence term. The purpose of the KL divergence term is to:**
- [ ] A) Ensure that the reconstructed output is sharp and realistic.
- [ ] B) Measure the difference between the input and the output.
- [ ] C) Act as a regularizer, forcing the learned latent distribution to be close to a standard normal distribution.
- [ ] D) Increase the capacity of the encoder.

**Correct Answer:** C

**Explanation:**
- C) The KL divergence term regularizes the latent space, ensuring that it is continuous and well-structured. This is what allows the VAE to be generative.

**16. The U-Net architecture is commonly used in which generative model?**
- [ ] A) Generative Adversarial Network (GAN)
- [ ] B) Variational Autoencoder (VAE)
- [ ] C) Diffusion Model
- [ ] D) Transformer

**Correct Answer:** C

**Explanation:**
- C) The U-Net architecture, with its skip connections between the downsampling and upsampling paths, is highly effective at predicting noise in an image while preserving fine details, making it the standard choice for the reverse process in Diffusion Models.

**17. What is the main difference between a standard Autoencoder (AE) and a Denoising Autoencoder (DAE)?**
- [ ] A) A DAE has more hidden layers than an AE.
- [ ] B) A DAE is trained to reconstruct a clean version of a corrupted input, forcing it to learn more robust features.
- [ ] C) A DAE uses a different loss function than an AE.
- [ ] D) A DAE can only be used for image data.

**Correct Answer:** B

**Explanation:**
- B) By training the model to remove noise, a DAE is forced to learn the underlying manifold of the data more effectively than a standard AE, which can sometimes just learn the identity function.

**18. The output gate in an LSTM controls:**
- [ ] A) Which information from the previous cell state to forget.
- [ ] B) Which new information to store in the cell state.
- [ ] C) Which parts of the cell state are used to compute the hidden state for the current time step.
- [ ] D) The learning rate of the LSTM cell.

**Correct Answer:** C

**Explanation:**
- C) The output gate takes the updated cell state, passes it through a tanh function, and then filters it to produce the new hidden state, which is the output for the current time step.

**19. The primary innovation of the Transformer architecture over sequence-to-sequence models with attention was:**
- [ ] A) The use of a more complex encoder-decoder structure.
- [ ] B) The complete reliance on self-attention mechanisms, removing the need for recurrent layers.
- [ ] C) The introduction of a new loss function for sequence modeling.
- [ ] D) The use of a larger vocabulary.

**Correct Answer:** B

**Explanation:**
- B) The paper "Attention Is All You Need" introduced an architecture that dispensed with recurrence entirely, relying solely on self-attention to capture dependencies within the sequence, which allowed for parallelization and superior performance.

**20. In a CNN, the purpose of a 1x1 convolution is often to:**
- [ ] A) Reduce the spatial dimensions of the feature maps.
- [ ] B) Act as a bottleneck layer to reduce the number of channels (depth) in the feature maps, thus reducing computation.
- [ ] C) Increase the receptive field of the network.
- [ ] D) Add padding to the input.

**Correct Answer:** B

**Explanation:**
- B) A 1x1 convolution can be used to change the depth of the volume. It is often used to reduce the number of channels before a more expensive 3x3 or 5x5 convolution, or to increase it after, as seen in bottleneck architectures like ResNet.

**21. The term "autoregressive" in the context of a Transformer decoder means that:**
- [ ] A) The decoder's predictions are independent of each other.
- [ ] B) The decoder predicts the entire output sequence in a single forward pass.
- [ ] C) The prediction for the current token is conditioned on the tokens that have been previously generated.
- [ ] D) The decoder uses a linear activation function.

**Correct Answer:** C

**Explanation:**
- C) Autoregressive models generate the output sequence one element at a time, and each new element is predicted based on the sequence of elements generated so far.

**22. A major drawback of Diffusion Models compared to GANs is their:**
- [ ] A) Lower image quality.
- [ ] B) Slower inference/generation speed due to the iterative denoising process.
- [ ] C) Unstable training dynamics.
- [ ] D) Inability to learn complex data distributions.

**Correct Answer:** B

**Explanation:**
- B) Generating a sample from a Diffusion Model requires a multi-step reverse process, which is significantly slower than the single forward pass required by a GAN generator.

**23. The cell state in an LSTM is often referred to as a "conveyor belt" because:**
- [ ] A) It moves information from the output layer back to the input layer.
- [ ] B) It allows information to flow through the network with only minor, controlled modifications.
- [ ] C) It is a very slow component of the LSTM cell.
- [ ] D) It can only store a single value at a time.

**Correct Answer:** B

**Explanation:**
- B) The cell state provides a direct path for information to be carried across many time steps. The gating mechanisms make small, controlled changes to this information, but it is largely preserved, which helps to mitigate the vanishing gradient problem.

**24. The main purpose of the feed-forward network (MLP) in each block of a Transformer is to:**
- [ ] A) Calculate the attention scores.
- [ ] B) Apply a non-linear transformation to the output of the attention layer.
- [ ] C) Combine the multiple attention heads.
- [ ] D) Normalize the layer's activations.

**Correct Answer:** B

**Explanation:**
- B) The self-attention layer itself is primarily a linear combination of value vectors. The MLP provides a crucial non-linear processing step, allowing the model to learn more complex functions.

**25. The loss function of a GAN is often described as a minimax game because:**
- [ ] A) The generator and discriminator are trying to cooperate to minimize the same loss function.
- [ ] B) The generator tries to minimize the discriminator's ability to distinguish real from fake, while the discriminator tries to maximize it.
- [ ] C) The loss function is always a minimum value.
- [ ] D) The loss function is always a maximum value.

**Correct Answer:** B

**Explanation:**
- B) It is a two-player game where the generator (G) tries to minimize the loss function (by fooling the discriminator), and the discriminator (D) tries to maximize it (by correctly identifying fakes). This is expressed as min_G max_D V(D, G).

**26. In a CNN, what is the effect of increasing the stride of a convolutional layer?**
- [ ] A) It increases the number of parameters in the layer.
- [ ] B) It reduces the spatial dimensions of the output feature map.
- [ ] C) It increases the receptive field of the neurons in the next layer.
- [ ] D) It has no effect on the output size.

**Correct Answer:** B

**Explanation:**
- B) A larger stride means the filter jumps more pixels at a time, resulting in a smaller output feature map. This is a form of downsampling.

**27. The input gate of an LSTM cell is responsible for:**
- [ ] A) Deciding which parts of the previous cell state to forget.
- [ ] B) Deciding which new information to store in the cell state.
- [ ] C) Filtering the cell state to produce the hidden state.
- [ ] D) Combining the hidden state and the cell state.

**Correct Answer:** B

**Explanation:**
- B) The input gate determines which new information from the current input and previous hidden state is important enough to be added to the long-term memory (the cell state).

**28. The primary advantage of subword tokenization (like WordPiece) over word-based tokenization is that it:**
- [ ] A) Creates a smaller vocabulary.
- [ ] B) Can handle out-of-vocabulary (OOV) words by breaking them into known sub-parts.
- [ ] C) Is computationally faster.
- [ ] D) Is easier to implement.

**Correct Answer:** B

**Explanation:**
- B) This is the key benefit. A word-based tokenizer fails on unseen words, but a subword tokenizer can represent any word by breaking it down, preserving its semantic meaning.

**29. The reconstruction loss of a VAE is often blurry compared to a GAN because:**
- [ ] A) The VAE decoder is less powerful than a GAN generator.
- [ ] B) The VAE is trained with a pixel-wise loss function (like MSE) that encourages the model to predict the average of possible outputs.
- [ ] C) The GAN's discriminator forces the generator to produce sharp images.
- [ ] D) Both B and C.

**Correct Answer:** D

**Explanation:**
- D) The blurriness of VAEs is a result of both the averaging nature of the reconstruction loss and the lack of an adversarial component to enforce realism and sharpness.

**30. The main purpose of skip connections in a U-Net architecture is to:**
- [ ] A) Reduce the number of layers in the network.
- [ ] B) Allow the upsampling path to directly access high-resolution features from the downsampling path, helping to preserve fine-grained details.
- [ ] C) Introduce more non-linearity into the network.
- [ ] D) Speed up the training process by skipping some layers.

**Correct Answer:** B

**Explanation:**
- B) The skip connections provide a direct path for high-resolution feature maps from the encoder to be concatenated with the upsampled feature maps in the decoder. This helps the decoder to reconstruct the image with much better detail.

**31. Which of the following is a key characteristic of a Transformer model?**
- [ ] A) It processes data sequentially, maintaining a hidden state at each time step.
- [ ] B) It relies on a self-attention mechanism to weigh the importance of different words in a sequence.
- [ ] C) It is primarily used for unsupervised learning tasks like clustering.
- [ ] D) It can only be used for text data.

**Correct Answer:** B

**Explanation:**
- B) The self-attention mechanism is the core innovation of the Transformer, allowing it to capture long-range dependencies and process sequences in parallel.

**32. The purpose of the `[CLS]` token in BERT is to:**
- [ ] A) Mark the end of a sentence.
- [ ] B) Act as a placeholder for masked words during pre-training.
- [ ] C) Provide an aggregate representation of the entire input sequence for classification tasks.
- [ ] D) Separate two sentences in the input.

**Correct Answer:** C

**Explanation:**
- C) The final hidden state corresponding to the `[CLS]` token is used as the input to a classifier for sentence-level tasks.

**33. In an LSTM, the sigmoid activation function is used in the gates to:**
- [ ] A) Scale the output to be between -1 and 1.
- [ ] B) Act as a switch, producing a value between 0 and 1 to control the flow of information.
- [ ] C) Introduce non-linearity to the cell state.
- [ ] D) Prevent the gradients from vanishing.

**Correct Answer:** B

**Explanation:**
- B) The output of the sigmoid function is interpreted as a percentage, allowing the gates to control how much information is forgotten, stored, or outputted.

**34. A key difference between a standard Autoencoder and a Variational Autoencoder (VAE) is that:**
- [ ] A) A VAE has a discriminator, while an AE does not.
- [ ] B) A VAE's encoder outputs a probability distribution (mean and variance) for the latent space, while an AE's encoder outputs a single point.
- [ ] C) A VAE is a supervised model, while an AE is unsupervised.
- [ ] D) A VAE cannot be used for dimensionality reduction.

**Correct Answer:** B

**Explanation:**
- B) This is the fundamental difference that makes VAEs generative. By learning a distribution, the latent space becomes continuous and structured, allowing for meaningful sampling.

**35. The primary goal of a Generative Adversarial Network (GAN) is to:**
- [ ] A) Learn a compressed representation of the input data.
- [ ] B) Classify data into multiple categories.
- [ ] C) Generate new data samples that are indistinguishable from a real dataset.
- [ ] D) Reconstruct a clean version of a noisy input.

**Correct Answer:** C

**Explanation:**
- C) GANs are generative models that learn to produce new, synthetic data that mimics the distribution of the training data.

**36. The forward process in a Diffusion Model is a:**
- [ ] A) Learned process that adds noise to an image.
- [ ] B) Fixed, non-learned process that gradually adds Gaussian noise to an image.
- [ ] C) Process that removes noise from an image.
- [ ] D) Process that is only used during inference.

**Correct Answer:** B

**Explanation:**
- B) The forward process is a simple Markov chain with fixed parameters that adds noise over a series of time steps. The model only learns the reverse process.

**37. The main advantage of a Convolutional Neural Network (CNN) for image processing is its ability to:**
- [ ] A) Process sequences of data.
- [ ] B) Learn spatial hierarchies of features, from simple edges to complex objects.
- [ ] C) Handle out-of-vocabulary words.
- [ ] D) Generate new images.

**Correct Answer:** B

**Explanation:**
- B) The use of convolutional filters allows CNNs to learn features in a hierarchical manner, which is highly effective for understanding the content of images.

**38. The role of the `tanh` activation function in an LSTM cell is often to:**
- [ ] A) Act as a gate to control information flow.
- [ ] B) Create new candidate values or scale the cell state to be between -1 and 1.
- [ ] C) Normalize the input data.
- [ ] D) Prevent the model from overfitting.

**Correct Answer:** B

**Explanation:**
- B) The `tanh` function is used to create the new candidate values in the input gate and to scale the cell state before it is passed to the output gate, keeping the values in a bounded range.

**39. The self-attention mechanism in a Transformer allows it to:**
- [ ] A) Process sequences of any length without memory constraints.
- [ ] B) Weigh the importance of all other words in a sequence when processing a specific word, regardless of their distance.
- [ ] C) Only attend to the previous word in the sequence.
- [ ] D) Only attend to the next word in the sequence.

**Correct Answer:** B

**Explanation:**
- B) This is the key advantage of self-attention over recurrence. It allows for direct connections between any two tokens, enabling the model to capture long-range dependencies effectively.

**40. The purpose of transposed convolutions in a Convolutional Autoencoder is to:**
- [ ] A) Downsample the feature maps in the encoder.
- [ ] B) Add non-linearity to the network.
- [ ] C) Upsample the feature maps in the decoder to reconstruct the image.
- [ ] D) Reduce the number of channels in the feature maps.

**Correct Answer:** C

**Explanation:**
- C) Transposed convolutions are a learnable upsampling method used in the decoder to increase the spatial dimensions of the feature maps and reconstruct the original image.

**41. The loss function of a GAN involves a minimax game between:**
- [ ] A) The encoder and the decoder.
- [ ] B) The generator and the discriminator.
- [ ] C) The input data and the output data.
- [ ] D) The training set and the test set.

**Correct Answer:** B

**Explanation:**
- B) The generator tries to minimize the loss by fooling the discriminator, while the discriminator tries to maximize it by correctly identifying fakes.

**42. The primary reason for the slow inference speed of Diffusion Models is:**
- [ ] A) The large number of parameters in the U-Net.
- [ ] B) The iterative, multi-step nature of the reverse denoising process.
- [ ] C) The need to compute the KL divergence at each step.
- [ ] D) The use of a complex loss function.

**Correct Answer:** B

**Explanation:**
- B) Generating an image requires hundreds or thousands of sequential passes through the denoising network, which is much slower than a single forward pass through a GAN or VAE.

**43. The input to a standard GAN generator is typically:**
- [ ] A) A real image from the training set.
- [ ] B) A vector of random noise.
- [ ] C) A compressed latent representation from an encoder.
- [ ] D) A class label.

**Correct Answer:** B

**Explanation:**
- B) The generator learns to transform a simple random noise vector into a complex, realistic data sample.

**44. The main purpose of the forget gate in an LSTM is to:**
- [ ] A) Decide which new information to add to the cell state.
- [ ] B) Control the flow of information from the cell state to the hidden state.
- [ ] C) Determine which parts of the long-term memory (cell state) are no longer needed and should be discarded.
- [ ] D) Reset the hidden state at each time step.

**Correct Answer:** C

**Explanation:**
- C) This gate is crucial for managing the long-term memory of the LSTM, allowing it to forget irrelevant information and retain important context.

**45. The key innovation of the Transformer architecture is its reliance on:**
- [ ] A) Recurrent layers.
- [ ] B) Convolutional layers.
- [ ] C) Self-attention mechanisms.
- [ ] D) Fully connected layers.

**Correct Answer:** C

**Explanation:**
- C) The Transformer was the first architecture to show that self-attention alone, without recurrence, is sufficient to achieve state-of-the-art performance on sequence transduction tasks.

**46. In a VAE, the latent space is regularized by forcing the learned distributions to be close to:**
- [ ] A) A uniform distribution.
- [ ] B) A standard normal distribution.
- [ ] C) The distribution of the input data.
- [ ] D) A discrete distribution.

**Correct Answer:** B

**Explanation:**
- B) The KL divergence term in the VAE loss function encourages the learned latent distributions to be close to a standard Gaussian, which results in a smooth and continuous latent space.

**47. The primary function of a pooling layer in a CNN is to:**
- [ ] A) Increase the number of feature maps.
- [ ] B) Introduce non-linearity.
- [ ] C) Reduce the spatial dimensions of the feature maps.
- [ ] D) Normalize the activations.

**Correct Answer:** C

**Explanation:**
- C) Pooling layers (like max pooling) downsample the feature maps, which reduces computational complexity and provides a degree of translational invariance.

**48. The main difference between a simple RNN and an LSTM is the presence of:**
- [ ] A) A hidden state.
- ] B) Shared weights.
- [ ] C) A cell state and gating mechanisms.
- [ ] D) An output layer.

**Correct Answer:** C

**Explanation:**
- C) The introduction of the cell state as a separate long-term memory channel, along with the forget, input, and output gates to control it, is the key architectural difference that allows LSTMs to overcome the vanishing gradient problem.

**49. The cross-attention layer in a Transformer decoder receives its Keys and Values from:**
- [ ] A) The decoder's previous layer.
- [ ] B) The encoder's output.
- [ ] C) A random noise vector.
- [ ] D) The target output sequence.

**Correct Answer:** B

**Explanation:**
- B) This is how the decoder gets context from the input sentence. The Queries come from the decoder, but the Keys and Values come from the encoder, allowing the decoder to attend to the most relevant parts of the source sequence.

**50. The primary advantage of a denoising autoencoder over a standard autoencoder is that it:**
- [ ] A) Is faster to train.
- [ ] B) Can be used for supervised learning tasks.
- [ ] C) Is forced to learn more robust features by learning to reconstruct a clean input from a corrupted one.
- [ ] D) Has a smaller bottleneck dimension.

**Correct Answer:** C

**Explanation:**
- C) By learning to remove noise, the denoising autoencoder is prevented from simply learning the identity function and is forced to capture the underlying manifold of the data more effectively.
