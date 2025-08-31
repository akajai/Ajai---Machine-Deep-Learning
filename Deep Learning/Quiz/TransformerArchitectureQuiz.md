# Transformer Architecture Quiz

**1. What is the primary reason Transformers can process sequences in parallel, unlike RNNs?**
- [ ] A) They use a more complex activation function.
- [ ] B) They require significantly less memory, allowing for parallel computation.
- [ ] C) The self-attention mechanism allows direct connections between any two tokens in the sequence, removing the need for sequential processing.
- [ ] D) They break the sentence into individual characters instead of words.

**Correct Answer:** C

**Explanation:**
- A) Incorrect. The activation functions used (like ReLU and Softmax) are standard and not the reason for parallelization.
- B) Incorrect. Transformers, especially large ones, are very memory-intensive; this is not an advantage they have over RNNs.
- C) Correct. RNNs must process token by token to maintain sequential context. The self-attention mechanism in Transformers calculates the relationships between all tokens simultaneously, eliminating this sequential dependency and enabling parallel processing.
- D) Incorrect. Tokenization strategy is separate from the core architectural difference that allows parallelization.

**2. In the self-attention mechanism, what are the three vectors created for each input token?**
- [ ] A) Word, Subword, and Character
- [ ] B) Query, Key, and Value
- [ ] C) Position, Embedding, and Context
- ] D) Input, Hidden, and Output

**Correct Answer:** B

**Explanation:**
- A) Incorrect. These are levels of tokenization, not the vectors used in self-attention.
- B) Correct. Each input token's embedding is multiplied by three distinct weight matrices (WQ, WK, WV) to produce a Query, a Key, and a Value vector, which are central to the attention calculation.
- C) Incorrect. Positional encodings and embeddings are inputs to the attention layer, not the specialized vectors created within it.
- D) Incorrect. These are general neural network concepts, not the specific vectors used in the self-attention calculation.

**3. How is the initial similarity between two tokens calculated in the scaled dot-product attention?**
- [ ] A) By taking the dot product of the current token's Query vector and the other token's Key vector.
- [ ] B) By passing their embeddings through a separate neural network.
- [ ] C) By measuring the Euclidean distance between their positional encodings.
- [ ] D) By taking the dot product of their Value vectors.

**Correct Answer:** A

**Explanation:**
- A) Correct. The dot product between a token's Query (its question) and another token's Key (its label) produces a raw similarity score, indicating their relevance to each other.
- B) Incorrect. While this is a valid approach in some attention mechanisms (additive attention), the Transformer uses scaled dot-product attention.
- C) Incorrect. Positional encodings are added to the embeddings but are not directly used for the similarity calculation in this manner.
- D) Incorrect. The Value vectors are used at the end of the calculation to create the output, not at the beginning to calculate similarity.

**4. What is the main purpose of the "mask" in the Masked Multi-Head Self-Attention layer of the Transformer decoder?**
- [ ] A) To hide the Keys and Values from the Query vectors.
- [ ] B) To prevent the decoder from "cheating" by attending to future tokens in the output sequence during training.
- [ ] C) To prevent the model from paying attention to padding tokens.
- [ ] D) To randomly deactivate neurons to prevent overfitting.

**Correct Answer:** B

**Explanation:**
- A) Incorrect. The mask is applied to the attention scores, not the Keys or Values directly.
- B) Correct. Since the decoder generates the output word by word (autoregressively), it should only be allowed to use the words it has already generated as context. The mask sets the attention scores for all future positions to negative infinity, so they become zero after the Softmax.
- C) Incorrect. While a padding mask is also used, the specific purpose of the mask in this decoder layer is to handle future tokens, not padding.
- D) Incorrect. This describes Dropout, which is a different regularization technique.

**5. Why is Positional Encoding a necessary component of the Transformer architecture?**
- [ ] A) To convert words into their subword tokens.
- [ ] B) To normalize the input values before they enter the encoder.
- [ ] C) Because the self-attention mechanism itself has no inherent sense of word order or position.
- [ ] D) To increase the dimensionality of the input embeddings.

**Correct Answer:** C

**Explanation:**
- A) Incorrect. This is the role of the tokenizer.
- B) Incorrect. This is the role of Layer Normalization.
- C) Correct. The self-attention mechanism treats the input as a "bag of words," calculating the relationship between every pair of tokens regardless of their position. Positional Encoding explicitly injects information about the position of each token into the model.
- D) Incorrect. While it is added to the embeddings, its purpose is to encode position, not just increase dimensionality.

**6. What is the primary function of the Cross-Attention layer in the Transformer decoder?**
- [ ] A) To combine the outputs of the multiple attention heads.
- [ ] B) To allow the decoder to attend to the output of the encoder, connecting the input and output sequences.
- [ ] C) To apply a non-linear transformation to the output of the masked attention layer.
- [ ] D) To allow the decoder to attend to the tokens it has previously generated.

**Correct Answer:** B

**Explanation:**
- A) Incorrect. This is done within each Multi-Head Attention block.
- B) Correct. In the Cross-Attention layer, the Queries come from the decoder, but the Keys and Values come from the encoder's output. This is the crucial step where the decoder looks at the input sentence to gather the context needed to generate the next word in the output sentence.
- C) Incorrect. This is the function of the Feed-Forward Network (MLP).
- D) Incorrect. This is the function of the Masked Self-Attention layer in the decoder.

**7. What is the main advantage of subword tokenization (e.g., WordPiece) over word-based tokenization?**
- [ ] A) It is computationally faster because it processes fewer tokens per sentence.
- [ ] B) It completely eliminates the out-of-vocabulary (OOV) problem by breaking unknown words into known sub-parts.
- [ ] C) It creates a much larger vocabulary, which is more expressive.
- [ ] D) It ensures that every word is mapped to a unique integer ID.

**Correct Answer:** B

**Explanation:**
- A) Incorrect. It often results in more tokens per sentence, not fewer.
- B) Correct. A word-based tokenizer maps any unseen word to a single `[UNK]` token, losing its meaning. A subword tokenizer can represent any word by breaking it down into smaller known pieces, thus preserving its semantic meaning.
- C) Incorrect. It creates a smaller, more efficient vocabulary.
- D) Incorrect. Word-based tokenization also does this, but only for words in its fixed vocabulary.

**8. In the analogy of self-attention as a library researcher, what do the "Value" vectors represent?**
- [ ] A) The library's card catalog.
- [ ] B) The title or label on the spine of a book.
- ] C) The actual content or information contained within a book.
- [ ] D) The researcher's specific question.

**Correct Answer:** C

**Explanation:**
- A) Incorrect. This analogy does not fit the Q, K, V framework.
- B) Incorrect. This is the Key.
- C) Correct. The Value vector represents the actual substance or information of a token, which is retrieved and blended into the final context vector based on the attention weights.
- D) Incorrect. This is the Query.

**9. What is the purpose of the Feed-Forward Network (MLP) sub-layer within each Transformer block?**
- [ ] A) To combine the multiple attention heads into a single vector.
- [ ] B) To apply a non-linear transformation, process the information from the attention layer, and add representational capacity.
- [ ] C) To normalize the outputs of the attention layer.
- [ ] D) To calculate the attention scores between tokens.

**Correct Answer:** B

**Explanation:**
- A) Incorrect. This is done by a linear projection after the multi-head attention calculation.
- B) Correct. The self-attention layer is primarily linear in how it combines value vectors. The MLP provides a crucial non-linear processing step, allowing the model to learn more complex functions.
- C) Incorrect. This is the role of Layer Normalization.
- D) Incorrect. This is done by the self-attention layer.

**10. How does Multi-Head Self-Attention (MHSA) enhance the Transformer's capabilities?**
- [ ] A) It replaces the need for positional encodings.
- [ ] B) It allows the model to focus on different types of relationships (e.g., grammatical, semantic) in parallel.
- [ ] C) It significantly reduces the number of parameters in the model.
- [ ] D) It allows the model to process multiple sentences at the same time.

**Correct Answer:** B

**Explanation:**
- A) Incorrect. Positional encodings are still required.
- B) Correct. By having multiple "heads," each with its own set of learned weights, the model can simultaneously attend to different aspects of the sequence. One head might learn to track subject-verb agreement while another tracks semantic similarity.
- C) Incorrect. It increases the number of parameters compared to single-head attention.
- D) Incorrect. Batching allows for processing multiple sentences, not MHSA itself.

**11. What is the out-of-vocabulary (OOV) problem?**
- [ ] A) When a word has multiple meanings (polysemy).
- [ ] B) When a tokenizer encounters a word during inference that was not in its training vocabulary.
- [ ] C) When a sentence is too long for the model to process.
- [ ] D) When a model produces a word that is not in the dictionary.

**Correct Answer:** B

**Explanation:**
- A) Incorrect. This is a problem of ambiguity, which attention helps to solve.
- B) Correct. This is the core definition of the OOV problem, where a word-based tokenizer has no representation for a new word and must map it to a generic `[UNK]` token, losing its meaning.
- C) Incorrect. This is a context length limitation, not OOV.
- D) Incorrect. This is a generation issue, not the OOV problem itself.

**12. Why are sinusoidal functions used for positional encodings in the original Transformer model?**
- [ ] A) Because they are the only functions that can represent position.
- [ ] B) Because they allow the model to easily learn relative positions and can generalize to sequences of unseen lengths.
- [ ] C) Because they are computationally cheaper than learned positional embeddings.
- [ ] D) Because they introduce randomness into the model.

**Correct Answer:** B

**Explanation:**
- A) Incorrect. Other methods, like learned embeddings, can also be used.
- B) Correct. The properties of sine and cosine functions mean that the encoding for a position `pos + k` can be represented as a linear function of the encoding at `pos`, making it easy for the model to understand relative positioning. They also don't have a fixed limit, unlike learned embeddings.
- C) Incorrect. While they are efficient, the primary reason is their ability to represent relative positions effectively.
- D) Incorrect. They are deterministic, not random.

**13. What is the role of the `[CLS]` token in a BERT-style Transformer model?**
- [ ] A) It is a padding token used to make all sequences in a batch the same length.
- [ ] B) Its final hidden state is used as the aggregate representation for the entire sequence for classification tasks.
- [ ] C) It marks the end of a sequence.
- ] D) It is used to separate two different sentences in the input.

**Correct Answer:** B

**Explanation:**
- A) Incorrect. This is the role of the `[PAD]` token.
- B) Correct. The `[CLS]` (classification) token is added to the beginning of every sequence, and its final hidden state is designed to be a summary of the whole sequence, which can be easily passed to a classifier.
- C) Incorrect. The `[SEP]` token often serves this purpose as well.
- D) Incorrect. This is the role of the `[SEP]` token.

**14. In the Transformer decoder, where do the Keys and Values for the Cross-Attention layer come from?**
- [ ] A) A separate, randomly initialized set of vectors.
- [ ] B) The output of the final encoder block.
- [ ] C) The target output sequence (the ground truth).
- [ ] D) The output of the previous masked self-attention layer in the decoder.

**Correct Answer:** B

**Explanation:**
- A) Incorrect. The Keys and Values must contain the contextual information from the input sentence.
- B) Correct. The Cross-Attention layer allows the decoder to look at the entire input sequence by using the encoder's output as the source for its Keys and Values, providing the necessary context for translation or generation.
- C) Incorrect. The model does not get to see the ground truth during this step.
- D) Incorrect. The Queries come from the decoder's previous layer, but not the Keys and Values.

**15. What is the purpose of the final Softmax layer in the Transformer decoder?**
- [ ] A) To combine the outputs of the multiple attention heads.
- [ ] B) To normalize the layer's activations to have a mean of zero.
- [ ] C) To convert the final output vector into a probability distribution over the entire vocabulary.
- [ ] D) To calculate the attention weights for the cross-attention layer.

**Correct Answer:** C

**Explanation:**
- A) Incorrect. This is done within the Multi-Head Attention block.
- B) Incorrect. This is the role of Layer Normalization.
- C) Correct. After the final decoder block, a linear layer projects the output into a vector the size of the vocabulary. The Softmax function then converts these raw scores (logits) into probabilities, so the model can predict the most likely next word.
- D) Incorrect. Softmax is used within the attention layers, but the final Softmax has a different purpose.

**16. What problem does the scaling factor ($$\frac{1}{\sqrt{d_k}}$$) in the scaled dot-product attention formula solve?**
- [ ] A) It reduces the computational cost of the matrix multiplication.
- [ ] B) It normalizes the Value vectors before they are summed.
- [ ] C) It ensures that the attention weights sum to 1.
- [ ] D) It prevents the dot products from growing too large in magnitude, which would push the Softmax function into regions with extremely small gradients.

**Correct Answer:** D

**Explanation:**
- A) Incorrect. It is an element-wise scaling and does not significantly change the computational cost.
- B) Incorrect. The Value vectors are not scaled in this way.
- C) Incorrect. The Softmax function itself ensures the weights sum to 1.
- D) Correct. For large values of the key dimension ($$d_k$$), the dot products can become very large. This can saturate the Softmax function, leading to vanishing gradients and hindering the learning process. The scaling factor counteracts this effect.

**17. How does the WordPiece algorithm decide which subwords to merge when building its vocabulary?**
- [ ] A) It merges tokens randomly to increase diversity.
- [ ] B) It merges the most frequent adjacent pairs of tokens in the corpus.
- [ ] C) It merges tokens based on their semantic similarity.
- [ ] D) It merges the longest possible subwords first.

**Correct Answer:** B

**Explanation:**
- A) Incorrect. The process is deterministic and based on frequency counts.
- B) Correct. The WordPiece algorithm is data-driven. It starts with individual characters and iteratively combines the most commonly occurring adjacent pairs to form new, longer subwords until the desired vocabulary size is reached.
- C) Incorrect. The merging is based on frequency, not a pre-defined notion of semantic similarity.
- D) Incorrect. The greedy approach is used when tokenizing a word, not when building the vocabulary.

**18. What is the role of a padding mask in Transformer training?**
- [ ] A) To mask out the Query vectors in the self-attention calculation.
- [ ] B) To ensure the model does not calculate loss or accuracy on the meaningless `[PAD]` tokens.
- [ ] C) To add extra padding to sentences that are too short.
- ] D) To prevent the model from attending to future positions in the decoder.

**Correct Answer:** B

**Explanation:**
- A) Incorrect. The mask is applied to the attention scores before the Softmax, not to the Query vectors.
- B) Correct. Since sentences in a batch are padded to the same length, the padding mask is used to tell the model which tokens are real and which are just padding, so that the padding is ignored during the loss and accuracy calculations.
- C) Incorrect. Padding is the process of adding the tokens; the mask is for ignoring them.
- D) Incorrect. This is the role of the look-ahead (or causal) mask in the decoder.

**19. Which component of the Transformer architecture is most directly responsible for solving the "information bottleneck" problem of RNNs?**
- [ ] A) The use of subword tokenization.
- [ ] B) The Positional Encoding.
- [ ] C) The Self-Attention mechanism.
- [ ] D) The Feed-Forward Network.

**Correct Answer:** C

**Explanation:**
- A) Incorrect. Tokenization solves the OOV problem, not the information bottleneck.
- B) Incorrect. Positional encoding solves the word order problem, not the information bottleneck.
- C) Correct. The self-attention mechanism creates direct paths between any two tokens in the sequence, regardless of their distance. This allows information to flow easily from the beginning to the end of a sequence, overcoming the bottleneck where RNNs have to pass information sequentially through every intermediate step.
- D) Incorrect. The MLP is for non-linear processing, not for solving the information bottleneck.

**20. In a machine translation task, the Cross-Attention layer helps the decoder to...**
- [ ] A) ...generate a vocabulary for the target language.
- [ ] B) ...learn the alignment between words in the source and target languages.
- [ ] C) ...process the source sentence in parallel.
- [ ] D) ...understand the grammar of the output language.

**Correct Answer:** B

**Explanation:**
- A) Incorrect. The vocabulary is predefined before training.
- B) Correct. By taking Queries from the generated output and Keys/Values from the source sentence, the cross-attention layer learns which words in the source are most relevant for predicting the next word in the target. This is effectively learning the alignment between the two languages.
- C) Incorrect. The encoder processes the source sentence in parallel; the decoder uses that output.
- D) Incorrect. This is learned more broadly by the entire decoder, particularly the masked self-attention.


### Back to Reading Content --> [Transformer Architecture](../TransformerArchitecture.md)