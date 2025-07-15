


# Embedding Layer
embedding_model = Word2VecEmbedding(tokens=tokenized_input, -----)
vocab_size = embedding_model.get_vocab_size()
word_vectors = embedding_model.get_word_vectors()

# Keras embedding layer initialized with the Word2Vec embeddings
embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=50,  # Match the vector_size in Word2VecEmbedding
        weights=[word_vectors],  # Initialize with Word2Vec weights
        trainable=True
    )
    
# Get the embeddings from the Keras layer
embeddings = embedding_layer(------)
