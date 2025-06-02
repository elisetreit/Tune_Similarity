class Embedder:
    def __init__(self, vocab_size, emb_dim, rnn_units):
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.rnn_units = rnn_units
        self.model = self.build_model()

    def build_model(self):
        from tensorflow.keras import layers, Model

        # Define your two inputs
        notes_in = layers.Input(shape=(None,), dtype="int32", name="note_ids")
        durs_in = layers.Input(shape=(None,), dtype="float32", name="durations")

        # Embed your notes (this is trainable, starts random)
        note_emb = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.emb_dim,
            mask_zero=True,
        )(notes_in)

        dur_feat = layers.Lambda(
            lambda x: tf.expand_dims(x, -1),
            mask=lambda inputs, mask: mask
        )(durs_in)
        
        # Project durations into EMB_DIM via a Dense layer
        dur_emb = layers.TimeDistributed(layers.Dense(self.emb_dim))(dur_feat)
        
        # Combine embeddings
        x = layers.Add()([note_emb, dur_emb])

        # Encode with a Bidirectional GRU
        rnn_out = layers.Bidirectional(
            layers.GRU(self.rnn_units, return_sequences=True)
        )(x)

        # Pool across time
        tune_vec = layers.GlobalAveragePooling1D()(rnn_out)

        # L2-normalize
        tune_emb = layers.Lambda(lambda z: tf.math.l2_normalize(z, axis=1))(tune_vec)

        # Build & compile
        model = Model(inputs=[notes_in, durs_in], outputs=tune_emb)
        model.compile(optimizer="adam", loss="mse")  # Placeholder loss

        return model

    def summary(self):
        return self.model.summary()