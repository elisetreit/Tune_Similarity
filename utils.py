import os
import json
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from pathlib import Path
from tensorflow.keras import layers, models
import collections
import random
import matplotlib.pyplot as plt

# ============================================================================

def load_model_from_experiment(experiment_name, vocab_size=None, checkpoint="best"):
    """Load model from experiment directory"""
    
    experiment_dir = Path(experiment_name)
    
    # Load config
    with open(experiment_dir / "config.json", 'r') as f:
        config = json.load(f)
    
    # Recreate model architecture
    if vocab_size is None:
        # Try to infer vocab size from config or data
        try:
            with open("tokenized_data/note_vocab.pkl", "rb") as f:
                vocab_list = pickle.load(f)
            vocab_size = len(vocab_list) + 2
        except:
            raise ValueError("Could not determine vocab_size. Please provide it explicitly.")
    
    model = create_tune_embedding_model(vocab_size, config)
    
    # Load weights
    if checkpoint == "best":
        weights_path = experiment_dir / "checkpoints" / "best_weights.h5"
    else:
        weights_path = experiment_dir / "checkpoints" / f"final_weights_{checkpoint}.h5"
    
    if weights_path.exists():
        model.load_weights(weights_path)
        print(f"✓ Loaded weights from: {weights_path}")
    else:
        print(f"⚠ Weights file not found: {weights_path}")
        return None
    
    return model, config

# ============================================================================
# SAVING AND LOADING FUNCTIONS
# ============================================================================

def save_model_and_weights(model, experiment_dir, experiment_name):
    """Save model in multiple formats for reliability"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save weights to experiment directory
    weights_path = experiment_dir / "checkpoints" / f"final_weights_{timestamp}.weights.h5"
    model.save_weights(weights_path)
    print(f"✓ Saved weights to: {weights_path}")
    
    # Save weights to main saved_models directory
    main_weights_path = Path("../saved_models") / f"{experiment_name}.weights.h5"
    model.save_weights(main_weights_path)
    print(f"✓ Saved weights to: {main_weights_path}")
    
    # Save model architecture
    architecture_path = experiment_dir / "model_architecture.json"
    with open(architecture_path, 'w') as f:
        f.write(model.to_json())
    print(f"✓ Saved architecture to: {architecture_path}")
    
    # Try to save full model (might fail with Lambda layers)
    try:
        model_path = experiment_dir / "checkpoints" / f"full_model_{timestamp}.keras"
        model.save(model_path)
        print(f"✓ Saved full model to: {model_path}")
    except Exception as e:
        print(f"⚠ Could not save full model: {e}")
    
    return weights_path, main_weights_path

class TuneSimilarityCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_df, eval_frequency=1):
        super().__init__()
        self.val_df = val_df
        self.eval_frequency = eval_frequency
        self.similarity_history = []
        
    def on_epoch_end(self, epoch, logs={}):
        if (epoch + 1) % self.eval_frequency == 0:
            # Run similarity evaluation
            metrics = evaluate_on_validation(self.model, self.val_df)
            
            # Log metrics
            logs.update({
                'val_pos_similarity': metrics['val_positive_similarity'],
                'val_neg_similarity': metrics['val_negative_similarity'],
                'val_separation': metrics['val_separation']
            })
            
            self.similarity_history.append(metrics)
            
            # Print metrics
            print(f"\nEpoch {epoch+1} Similarity Metrics:")
            print(f"Positive: {metrics['val_positive_similarity']:.3f}")
            print(f"Negative: {metrics['val_negative_similarity']:.3f}")
            print(f"Separation: {metrics['val_separation']:.3f}\n")

# ============================================================================
# UPDATED EVALUATION FUNCTIONS
# ============================================================================
def create_evaluation_pairs(df, num_positive=500, num_negative=1000):
    """Create evaluation pairs for validation"""
    
    # Filter tunes with multiple settings
    multi_setting_tunes = df.tune_id.value_counts()
    multi_setting_tunes = multi_setting_tunes[multi_setting_tunes >= 2].index
    
    if len(multi_setting_tunes) == 0:
        return [], []
    
    # Filter DataFrame to only include multi-setting tunes
    df_filtered = df[df.tune_id.isin(multi_setting_tunes)]
    
    # Create positive pairs (same tune, different settings)
    pos_pairs = []
    for tune_id in multi_setting_tunes:
        indices = df_filtered[df_filtered.tune_id == tune_id].index.tolist()
        if len(indices) < 2:
            continue
        random.shuffle(indices)
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                pos_pairs.append((indices[i], indices[j]))
                if len(pos_pairs) >= num_positive:
                    break
            if len(pos_pairs) >= num_positive:
                break
        if len(pos_pairs) >= num_positive:
            break
    
    # Create negative pairs (different tunes)
    neg_pairs = []
    all_indices = df_filtered.index.tolist()
    random.shuffle(all_indices)
    
    for i in range(len(all_indices)):
        for j in range(i + 1, len(all_indices)):
            if df_filtered.iloc[all_indices[i]].tune_id != df_filtered.iloc[all_indices[j]].tune_id:
                neg_pairs.append((all_indices[i], all_indices[j]))
                if len(neg_pairs) >= num_negative:
                    break
        if len(neg_pairs) >= num_negative:
            break
    
    return pos_pairs, neg_pairs

def evaluate_on_validation(model, val_df):
    """Quick evaluation on validation set during training"""
    
    if len(val_df) == 0:
        return {}
    
    # Create evaluation pairs
    pos_pairs, neg_pairs = create_evaluation_pairs(
        val_df, num_positive=min(500, len(val_df)//4), num_negative=min(1000, len(val_df)//2)
    )
    
    if len(pos_pairs) == 0:
        return {"note": "No evaluation possible - insufficient multi-setting tunes"}
    
    # Get embeddings for all required indices
    all_indices = list(set([i for pair in pos_pairs + neg_pairs for i in pair]))
    all_embeddings = {}
    
    for idx in all_indices:
        notes = val_df.iloc[idx].note_ids
        durs = val_df.iloc[idx].dur_seq
        embedding = model.predict([
            np.array([notes]), 
            np.array([durs])
        ], verbose=0)[0]
        all_embeddings[idx] = embedding
    
    # Calculate similarities
    pos_sims = [
        np.dot(all_embeddings[i], all_embeddings[j])
        for i, j in pos_pairs
    ]
    
    neg_sims = [
        np.dot(all_embeddings[i], all_embeddings[j])
        for i, j in neg_pairs
    ]
    
    return {
        "val_positive_similarity": np.mean(pos_sims),
        "val_negative_similarity": np.mean(neg_sims),
        "val_separation": np.mean(pos_sims) - np.mean(neg_sims)
    }

def create_training_dataset_with_validation(train_df, val_df, config):
    """Create both training and validation datasets"""
    
    training_params = config["training_params"]
    
    # Prepare training data (same as before but using train_df)
    by_id = collections.defaultdict(list)

    for notes, durs, tid in zip(train_df.note_ids, train_df.dur_seq, train_df.tune_id):
        by_id[int(tid)].append((notes, durs))
    
    tune_ids = list(by_id.keys())
    
    def balanced_sample_generator():
        while True:
            chosen = random.sample(tune_ids, training_params["batch_tunes"])
            for tid in chosen:
                examples = random.choices(by_id[tid], k=training_params["per_tune"])
                for notes, durs in examples:
                    yield (notes, durs), tid
    
    # Training dataset
    train_ds = tf.data.Dataset.from_generator(
        balanced_sample_generator,
        output_signature=(
            (tf.TensorSpec(shape=(None,), dtype=tf.int32),
             tf.TensorSpec(shape=(None,), dtype=tf.float32)),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    ).padded_batch(
        batch_size=training_params["batch_size"],
        padded_shapes=(([None], [None]), []),
        padding_values=((0, 0.0), 0)
    ).prefetch(tf.data.AUTOTUNE)
    
    # Validation dataset (if val_df has multi-setting tunes)
    val_ds = None
    val_multi_setting = val_df.tune_id.value_counts()
    val_multi_setting_tunes = val_multi_setting[val_multi_setting >= 2].index
    
    if len(val_multi_setting_tunes) > 0:
        val_df_filtered = val_df[val_df.tune_id.isin(val_multi_setting_tunes)]
        
        val_by_id = collections.defaultdict(list)
        for notes, durs, tid in zip(val_df_filtered.note_ids, val_df_filtered.dur_seq, val_df_filtered.tune_id):
            val_by_id[int(tid)].append((notes, durs))
        
        val_tune_ids = list(val_by_id.keys())
        
        def val_generator():
            # Generate a fixed set for validation (not infinite)
            samples = []
            for tid in val_tune_ids:
                examples = random.choices(val_by_id[tid], k=2)  # 2 examples per tune
                for notes, durs in examples:
                    samples.append(((notes, durs), tid))
            
            # Shuffle once
            random.shuffle(samples)
            for sample in samples:
                yield sample
        
        val_ds = tf.data.Dataset.from_generator(
            val_generator,
            output_signature=(
                (tf.TensorSpec(shape=(None,), dtype=tf.int32),
                 tf.TensorSpec(shape=(None,), dtype=tf.float32)),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        ).padded_batch(
            batch_size=training_params["batch_size"],
            padded_shapes=(([None], [None]), []),
            padding_values=((0, 0.0), 0)
        ).prefetch(tf.data.AUTOTUNE)
        
        print(f"✓ Validation dataset created with {len(val_tune_ids)} tunes")
    else:
        print("⚠️  No multi-setting tunes in validation set - skipping validation dataset")
    
    return train_ds, val_ds

def make_balanced_ds(df_split, batch_tunes=16, per_tune=2):
    # Build index: tune_id → list of (notes, durs)
    by_id = collections.defaultdict(list)
    for notes, durs, tid in zip(df_split.note_ids, df_split.dur_seq, df_split.tune_id):
        by_id[int(tid)].append((notes, durs))

    # Only include tunes with at least `per_tune` settings
    eligible_tune_ids = [tid for tid, settings in by_id.items() if len(settings) >= per_tune]

    def gen():
        while True:
            # Randomly sample `batch_tunes` *unique* tune IDs
            chosen_tunes = random.sample(eligible_tune_ids, batch_tunes)
            for tid in chosen_tunes:
                # Sample `per_tune` *unique* settings for this tune
                examples = random.sample(by_id[tid], per_tune)
                for notes, durs in examples:
                    yield (notes, durs), tid

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            (
                tf.TensorSpec(shape=(None,), dtype=tf.int32),   # notes_list
                tf.TensorSpec(shape=(None,), dtype=tf.float32)  # durs_list
            ),
            tf.TensorSpec(shape=(), dtype=tf.int32)            # label
        )
    ).padded_batch(
        batch_size=batch_tunes * per_tune,
        padded_shapes=(
            ([None], [None]),  # notes and durs
            []                 # label
        ),
        padding_values=(
            (0, 0.0),          # pad notes and durs
            0                  # pad label
        )
    ).prefetch(tf.data.AUTOTUNE)

    return ds





# ============================================================================
# DATA LOADING
# ============================================================================

def load_training_data():
    """Load training and validation data for experiments"""
    
    print("📁 Loading training data...")
    
    # Load splits
    train_df = pd.read_pickle('../tokenized_data/train_dataset.pkl')
    val_df = pd.read_pickle('../tokenized_data/val_dataset.pkl')
    
    # Load vocabulary
    with open("../tokenized_data/note_vocab.pkl", "rb") as f:
        vocab_list = pickle.load(f)
    
    # Load split metadata for reference
    with open("../tokenized_data/split_metadata.json", "r") as f:
        split_info = json.load(f)
    
    print(f"✓ Train: {len(train_df):,} samples from {train_df.tune_id.nunique():,} tunes")
    print(f"✓ Val:   {len(val_df):,} samples from {val_df.tune_id.nunique():,} tunes")
    print(f"✓ Vocab: {len(vocab_list)} unique notes")
    
    return train_df, val_df, vocab_list, split_info

def load_test_data():
    """Load test data (only for final evaluation!)"""
    
    print("🔒 Loading test data for final evaluation...")
    
    test_df = pd.read_pickle('../tokenized_data/test_dataset.pkl')
    
    with open("../tokenized_data/note_vocab.pkl", "rb") as f:
        vocab_list = pickle.load(f)
    
    print(f"✓ Test: {len(test_df):,} samples from {test_df.tune_id.nunique():,} tunes")
    print("⚠️  Remember: Use test set only for final model comparison!")
    
    return test_df, vocab_list

# ============================================================================
# LOSS FUNCTION
# ============================================================================

def batch_hard_triplet_loss(margin=0.3):
    """Batch hard triplet loss"""
    def loss_fn(y_true, y_pred):
        labels = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        embeddings = y_pred
        
        # Pairwise distances
        dot = tf.matmul(embeddings, embeddings, transpose_b=True)
        sq = tf.reduce_sum(tf.square(embeddings), axis=1, keepdims=True)
        pdist = tf.maximum(sq - 2.0 * dot + tf.transpose(sq), 0.0)
        
        # Masks
        labels_eq = tf.equal(tf.expand_dims(labels,1), tf.expand_dims(labels,0))
        mask_pos = tf.cast(labels_eq, tf.float32) - tf.eye(tf.shape(labels)[0])
        mask_neg = 1.0 - tf.cast(labels_eq, tf.float32)

        # Hard examples
        hardest_pos = tf.reduce_max(pdist * mask_pos, axis=1)
        max_dist = tf.reduce_max(pdist)
        pdist_neg = pdist + max_dist * (1.0 - mask_neg)
        hardest_neg = tf.reduce_min(pdist_neg, axis=1)

        # Triplet loss
        tl = tf.maximum(hardest_pos - hardest_neg + margin, 0.0)
        return tf.reduce_mean(tl)
    
    return loss_fn

# ============================================================================
# DIRECTORY SETUP
# ============================================================================

def setup_experiment_directories(experiment_name):
    """Create directory structure for experiment"""
    
    base_dir = Path("..")
    experiment_dir = base_dir / experiment_name
    
    # Create directories
    dirs_to_create = [
        experiment_dir,
        experiment_dir / "checkpoints",
        experiment_dir / "logs",
        experiment_dir / "plots",
        base_dir / "saved_models",
        base_dir / "tokenized_data"
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")
    
    return experiment_dir


def plot_training_val_loss(history):
    # history is the History object returned by model.fit(...)
    train_loss = history.history['loss']
    val_loss   = history.history.get('val_loss', None)
    epochs     = range(1, len(train_loss) + 1)

    plt.figure(figsize=(8,4))
    plt.plot(epochs, train_loss,  label='Train Loss')
    if val_loss is not None:
        plt.plot(epochs, val_loss,  label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()