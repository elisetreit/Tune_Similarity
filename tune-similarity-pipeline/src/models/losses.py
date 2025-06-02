def triplet_loss(y_true, y_pred, margin=0.3):
    # y_true: [batch] int labels (tune_id)
    # y_pred: [batch, dim] embeddings (already ℓ₂‐normalized or not)
    labels = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    embeddings = y_pred

    # 1) Pairwise distance matrix
    pdist = pairwise_distances(embeddings)

    # 2) Masks for positive / negative pairs
    labels_eq = tf.equal(tf.expand_dims(labels, 1), tf.expand_dims(labels, 0))  # [B,B]
    mask_pos = tf.cast(labels_eq, tf.float32) - tf.eye(tf.shape(labels)[0])   # zero diagonal
    mask_neg = 1.0 - tf.cast(labels_eq, tf.float32)

    # 3) For each anchor i, hardest positive = max_{j!=i, same label} d(i,j)
    hardest_pos = tf.reduce_max(pdist * mask_pos, axis=1)

    # 4) For each anchor i, easiest negative = min_{k, different label} d(i,k)
    #    to do that, add large constant to positives so they’re ignored in min()
    max_dist = tf.reduce_max(pdist)
    pdist_neg = pdist + max_dist * (1.0 - mask_neg)
    hardest_neg = tf.reduce_min(pdist_neg, axis=1)

    # 5) Combine with margin
    tl = tf.maximum(hardest_pos - hardest_neg + margin, 0.0)
    return tf.reduce_mean(tl)