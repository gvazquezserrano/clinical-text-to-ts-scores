from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import RandomOverSampler
from threading import Thread, Lock
from contextlib import redirect_stdout
from os.path import exists
import tensorflow as tf
import numpy as np
import pickle
import copy
import io
import math

# Import necessary global variables
from config.config import *


#--------------------------------------------------------------------------------#
# Auxiliary functions                                                            #
#--------------------------------------------------------------------------------#

def windowed_range(n):
    labels = np.arange(n, dtype=np.float32) / float(n-1)
    thresholds = [(labels[i] + labels[i+1]) / 2. for i in range(n-1)]

    t_start = [0.]
    t_start.extend(thresholds)
    #print(f'Thresholds start:', t_start)

    t_stop = thresholds
    t_stop.append(1.)
    #print(f'Thresholds stop:', t_stop)
    
    for i in range(n):
        yield t_start[i], t_stop[i], i


def enumerate_predictions(predictions,  n_labels=2, fold=None, client=None):
    one_hot = np.eye(n_labels)

    # filter fold an client:
    if fold==None:
        y = predictions[:, :] if client==None else predictions[:, client]
    else:
        y = predictions[fold, :] if client==None else predictions[fold, client]

    # reshape to two columns:                        
    y = y.reshape((-1, 2))

    # filter NaN-values:
    y = y[~np.isnan(y).any(axis=1), :]

    for y_true, y_pred in y:
        # create labels:
        l = 0
        for start, stop, i in windowed_range(n_labels):
            if (y_true > start) & (y_true <= stop):
                l = i
                break

        # calculate error:
        e = np.abs(np.arange(n_labels, dtype=np.float32) - (float(n_labels-1) * y_pred))
        e = np.clip(e, 0., 1.)

        # Yield (y_true, y_pred)
        yield one_hot[l], 1-e


def stratified_split(data, n_splits, stratify=None, shuffle=False, random_state=None):
    result = None
    if (n_splits > 1): 
        i_split, i_remainder = train_test_split(
            range(len(data)),
            train_size=(1.0 / float(n_splits)),
            stratify=stratify,
            shuffle=shuffle,
            random_state=random_state
        )
        result = stratified_split(
            data[i_remainder],
            n_splits - 1, 
            stratify=None if np.any(stratify==None) else stratify[i_remainder],
            shuffle=shuffle,
            random_state=random_state
        )
        result.append(data[i_split])

    else:
        result = [data]

    return result


#--------------------------------------------------------------------------------#
# Backbone-related classes and functions                                         #
#--------------------------------------------------------------------------------#

# Sinusoidal positional encoding for transformer backbone
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)

    def call(self, inputs):
        # inputs: (batch, seq_len, d_model)
        seq_len = tf.shape(inputs)[1]
        # Prefer static last dim when available
        d_model_static = inputs.shape[-1]
        if d_model_static is not None:
            d_model = int(d_model_static)
        else:
            d_model = tf.shape(inputs)[2]  # dynamic fallback

        # Create position indices
        position = tf.cast(tf.range(seq_len)[:, tf.newaxis], dtype=tf.float32)  # (seq_len, 1)

        # Create div_term for even indices
        # Use python range with d_model as int when available
        even_indexes = tf.range(0, d_model, 2)
        div_term = tf.exp(tf.cast(even_indexes, tf.float32) * -(tf.math.log(10000.0) / tf.cast(d_model, tf.float32)))
        # shape (d_model//2,)

        # Compute sin / cos
        sin_part = tf.sin(position * div_term)  # (seq_len, d_model//2)
        cos_part = tf.cos(position * div_term)  # (seq_len, d_model//2)

        pos_encoding = tf.concat([sin_part, cos_part], axis=-1)  # (seq_len, maybe d_model or d_model+1)
        pos_encoding = pos_encoding[:, :d_model]  # ensure exact width

        # expand to batch dim and add
        pos_encoding = pos_encoding[tf.newaxis, ...]  # (1, seq_len, d_model)
        return inputs + tf.cast(pos_encoding, inputs.dtype)


def transformer_encoder_block(inputs, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1, name_prefix="", attention_mask=None):
    """
    Creates a Transformer Encoder block.
    
    Args:
        inputs: Input tensor (shape: [batch_size, sequence_length, embed_dim])
        embed_dim: The dimensionality of the embedding (e.g., 16)
        num_heads: Number of attention heads (e.g., 4)
        feed_forward_dim: Internal dimension of the FFN (e.g., 32)
        name_prefix: String to prepend to layer names.
        
    Returns:
        A tensor of the same shape as inputs.
    """
    
    # 1. Layer Norm → Multi-Head Attention
    normed_inputs = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, 
        name=f'{name_prefix}_PreNorm1'
    )(inputs)
    
    attention_output = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embed_dim // num_heads,
        name=f'{name_prefix}_MHA'
    )(query=normed_inputs, value=normed_inputs, key=normed_inputs, attention_mask=attention_mask)
    
    attention_output = tf.keras.layers.Dropout(dropout_rate, name=f'{name_prefix}_MHA_Dropout')(attention_output)
    
    # Residual connection
    add_1 = inputs + attention_output
    
    # 2. Layer Norm → FFN
    normed_add1 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, 
        name=f'{name_prefix}_PreNorm2'
    )(add_1)
    
    ffn = tf.keras.Sequential([
        tf.keras.layers.Dense(feed_forward_dim, activation="relu", name=f'{name_prefix}_FFN_Dense1'),
        tf.keras.layers.Dense(embed_dim, name=f'{name_prefix}_FFN_Dense2'),
    ], name=f'{name_prefix}_FFN')
    ffn_output = ffn(normed_add1)
    
    ffn_output = tf.keras.layers.Dropout(dropout_rate, name=f'{name_prefix}_FFN_Dropout')(ffn_output)
    
    # Residual connection
    add_2 = add_1 + ffn_output
    
    return add_2


# Create the backbone model stack
def create_model_stack(input_layer, model_type, name_prefix):
    """
    Builds the sequential processing stack (GRU, LSTM, or Transformer).
    
    Args:
        input_layer: The masked input tensor.
        model_type: The string "GRU", "LSTM", or "transformer".
        name_prefix: String to prepend to layer names (e.g., 'vitals', 'labs').
        
    Returns:
        A tensor representing the final summary vector (shape [batch_size, RNN_DIM]).
    """
    
    if model_type == "GRU":
        # This is the original GRU code from Randl et al.
        layer1 = tf.keras.layers.GRU(RNN_DIM, return_sequences=True, name=f'GRU_layer1_{name_prefix}')(input_layer)
        layer2 = tf.keras.layers.GRU(RNN_DIM, return_sequences=True, name=f'GRU_layer2_{name_prefix}')(layer1)
        layer3 = tf.keras.layers.GRU(RNN_DIM, return_sequences=False, name=f'GRU_layer3_{name_prefix}')(layer2)
        return layer3

    elif model_type == "LSTM":
        # This is the LSTM modification. We just swap 'GRU' for 'LSTM'. All parameters are the same.
        layer1 = tf.keras.layers.LSTM(RNN_DIM, return_sequences=True, name=f'LSTM_layer1_{name_prefix}')(input_layer)
        layer2 = tf.keras.layers.LSTM(RNN_DIM, return_sequences=True, name=f'LSTM_layer2_{name_prefix}')(layer1)
        layer3 = tf.keras.layers.LSTM(RNN_DIM, return_sequences=False, name=f'LSTM_layer3_{name_prefix}')(layer2)
        return layer3

    elif model_type == "transformer":
        # This is a Transformer modification
        embed_dim_internal = EMBED_DIM_INTERNAL   
        feed_forward_dim = FEED_FORWARD_DIM   
        num_heads = NUM_HEADS 
        dropout_rate = DROPOUT_RATE
        num_transformer_blocks = NUM_TRANSFORMER_BLOCKS 

        # Project input features
        projection = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(embed_dim_internal),
            name=f'Input_Proj_{name_prefix}'
        )(input_layer)

        # Scale embeddings before Positional Encoding 
        # (standard Transformer practice to balance signal variance)
        projection = projection * tf.math.sqrt(tf.cast(embed_dim_internal, tf.float32))

        # Add sinusoidal positional encodings
        projection = PositionalEncoding(name=f'Pos_Encoding_{name_prefix}')(projection)
        
        # Build masks
        # mask_bool: shape (batch, seq_len) - True if valid, False if padding
        mask_bool = tf.logical_not(tf.reduce_all(tf.equal(input_layer, -2.0), axis=-1))
        # attn_mask: shape (batch, 1, seq_len) for MHA
        attn_mask = tf.expand_dims(mask_bool, axis=1) 
        
        # explicit attn mask: (batch, seq_len, seq_len)
        attn_mask = tf.tile(attn_mask, [1, tf.shape(input_layer)[1], 1])   # (batch, seq_len, seq_len)
        # or equivalently:
        # attn_mask = tf.cast(tf.matmul(tf.expand_dims(mask_bool,2), tf.expand_dims(mask_bool,1)), tf.bool)
        
        # Transformer Encoder Blocks
        transformer_output = projection
        for block_idx in range(num_transformer_blocks):
            transformer_output = transformer_encoder_block(
                inputs=transformer_output,
                embed_dim=embed_dim_internal, 
                num_heads=num_heads, 
                feed_forward_dim=feed_forward_dim,
                dropout_rate=dropout_rate,
                name_prefix=f'Transformer_Block_{block_idx + 1}_{name_prefix}',
                attention_mask=attn_mask
            )
        
        # Global Average Pooling (instead of last timestep, which is optimal for RNNs) 
        # Instead of taking the last step, we average all valid steps.
        # This allows the model to aggregate information from the whole sequence.
        
        # Expand mask to match output dim: (batch, seq_len, embed_dim)
        mask_float = tf.cast(mask_bool, dtype=transformer_output.dtype)
        mask_expanded = tf.expand_dims(mask_float, axis=-1)
        
        # Zero out the padding locations in the output
        transformer_output_masked = transformer_output * mask_expanded
        
        # Sum over the sequence length
        sum_output = tf.reduce_sum(transformer_output_masked, axis=1)
        
        # Count valid steps (avoid division by zero)
        sum_mask = tf.reduce_sum(mask_float, axis=1)
        sum_mask = tf.maximum(sum_mask, 1.0) # prevent div by zero
        sum_mask = tf.expand_dims(sum_mask, axis=-1)
        
        # Calculate Mean
        pooled_output = sum_output / sum_mask

        # Project back down to RNN_DIM to keep output consistent (in case embed_dim_internal =! RNN_DIM)
        if embed_dim_internal != RNN_DIM:
            pooled_output = tf.keras.layers.Dense(16, activation='relu', name=f'ProjDown_{name_prefix}')(pooled_output)
        
        return pooled_output    
    
    else:
        raise ValueError(f"Unknown MODEL_ARCHITECTURE: {model_type}. Must be 'GRU', 'LSTM', or 'transformer'.")


#--------------------------------------------------------------------------------#
# Training setup for transformer model                                           #
#--------------------------------------------------------------------------------#

class WarmUpCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Linear warmup to peak_lr for `warmup_steps` steps, then cosine decay to final_lr
    over the remaining `decay_steps`.
    """
    def __init__(self, peak_lr, warmup_steps, decay_steps, final_lr=0.0, name=None):
        super().__init__()
        self.peak_lr = tf.convert_to_tensor(peak_lr, dtype=tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.decay_steps = tf.cast(decay_steps, tf.float32)
        self.final_lr = tf.convert_to_tensor(final_lr, dtype=tf.float32)
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "WarmUpCosineDecay"):
            step = tf.cast(step, tf.float32)

            # linear warmup
            warmup_lr = self.peak_lr * (step / tf.maximum(1.0, self.warmup_steps))

            # cosine decay (from peak_lr to final_lr)
            decay_step = tf.maximum(0.0, step - self.warmup_steps)
            cosine_decay = 0.5 * (1.0 + tf.cos(math.pi * tf.minimum(decay_step, self.decay_steps) / tf.maximum(1.0, self.decay_steps)))
            decayed = (self.peak_lr - self.final_lr) * cosine_decay + self.final_lr

            lr = tf.where(step < self.warmup_steps, warmup_lr, decayed)
            return lr

    def get_config(self):
        return {
            "peak_lr": float(self.peak_lr.numpy()) if hasattr(self.peak_lr, "numpy") else float(self.peak_lr),
            "warmup_steps": int(self.warmup_steps),
            "decay_steps": int(self.decay_steps),
            "final_lr": float(self.final_lr.numpy()) if hasattr(self.final_lr, "numpy") else float(self.final_lr),
            "name": self.name
        }

if MODEL_ARCHITECTURE == "transformer":
              
    num_epochs = NUM_EPOCHS               
    peak_lr = PEAK_LR                 
    final_lr = FINAL_LR
    weight_decay = WEIGHT_DECAY            
    clip_norm = CLIP_NORM                
    steps_per_epoch = STEPS_PER_EPOCH
    
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = min(1000, max(50, total_steps // 10))  # warmup = 10% of training or up to 1000 steps
    decay_steps = max(1, total_steps - warmup_steps)
    
    lr_schedule = WarmUpCosineDecay(
        peak_lr=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        final_lr=final_lr
    )
    

#--------------------------------------------------------------------------------#
# Data builder                                                                   #
#--------------------------------------------------------------------------------#

class DataBuilder:
    def __init__(self, icustays, vitals, labs, notes=None, key='icustay_id', random_state=None):
        '''Creates a new DataBuilder Object and initializes normalization based on a list of ICU-stays.

        PARAMETERS
            icustays (np.array):                Array containing icustay-ids and labels

            vitals (pd.DataFrameGroupBy):       Vitals grouped by key

            labs (pd.DataFrameGroupBy):         Labs grouped by key

            notes (pd.DataFrameGroupBy):        Notes grouped by key (Optional, only used if ADD_NOTES=True)
            
            key (string):                       Name of the column to be used as key (default: 'icustay_id')

            random_state (int):                 Seed for the random generator (default: None)
        '''
        # Only possible if icustays not empty:
        if len(icustays) == 0:
            raise Exception("Paramter 'icustays' must be a non-empty list.")

        # Save properties:
        self.vitals = vitals if ADD_VITALS else None
        self.labs = labs if ADD_LABS else None
        self.notes = notes if ADD_NOTES else None

        self.key = key
        self.random_state = random_state

        # Init iteration variables:
        max_vitals = None; min_vitals = None
        max_labs = None;   min_labs = None
        max_notes = None;  min_notes = None

        # Find min and max for vitals, labs and notes:
        for X, y in self.__icustays2data(icustays):
            idx = 0
            if ADD_VITALS:
                act_max = np.max(X[idx], axis=0)
                act_min = np.min(np.where(X[idx] < 0, float('inf'), X[idx]), axis=0)
                max_vitals = act_max if max_vitals is None else np.maximum(act_max, max_vitals)
                min_vitals = act_min if min_vitals is None else np.minimum(act_min, min_vitals)
                idx += 1
            if ADD_LABS:
                act_max = np.max(X[idx], axis=0)
                act_min = np.min(np.where(X[idx] < 0, float('inf'), X[idx]), axis=0)
                max_labs = act_max if max_labs is None else np.maximum(act_max, max_labs)
                min_labs = act_min if min_labs is None else np.minimum(act_min, min_labs)
                idx += 1
            if ADD_NOTES:
                act_max = np.max(X[idx], axis=0)
                act_min = np.min(np.where(X[idx] < 0, float('inf'), X[idx]), axis=0)
                max_notes = act_max if max_notes is None else np.maximum(act_max, max_notes)
                min_notes = act_min if min_notes is None else np.minimum(act_min, min_notes)

        # Convert to tensorflow for better performance:
        if ADD_VITALS:
            self.max_vitals = tf.constant(max_vitals, dtype=tf.float64)
            self.min_vitals = tf.constant(min_vitals, dtype=tf.float64)
        if ADD_LABS:
            self.max_labs = tf.constant(max_labs, dtype=tf.float64)
            self.min_labs = tf.constant(min_labs, dtype=tf.float64)
        if ADD_NOTES:
            self.max_notes = tf.constant(max_notes, dtype=tf.float64)
            self.min_notes = tf.constant(min_notes, dtype=tf.float64)

    def __icustays2data(self, icustays):
        '''Generator, that yields the data matching a list of ICU-stays.

        PARAMETERS
            icustays (np.array):                Array containing icustay-ids and labels
            
        YIELDS
            A tuple (X_streams, y) of data X and label y
        '''

        flatten = tf.keras.layers.Flatten()

        for icustay, label in icustays:
            # Extract input streams:
            X_streams = []
            if ADD_VITALS:
                X_streams.append(flatten(self.vitals.get_group(icustay).drop(self.key, axis=1).to_numpy()))
            if ADD_LABS:
                X_streams.append(flatten(self.labs.get_group(icustay).drop(self.key, axis=1).to_numpy()))
            if ADD_NOTES:
                X_streams.append(flatten(self.notes.get_group(icustay).drop(self.key, axis=1).to_numpy()))

            # Create label-tensor:
            y = tf.constant(label, dtype=tf.dtypes.float64, shape=1)

            yield tuple(X_streams), y

    def __normalize(self, X, y):
        '''Performs normalizazion on each sample.

        PARAMETERS
            X_streams (tensor):         Data sample in the form of a tuple (vitals, labs, notes)
            y (tensor):                 Data label

        RETURNS
            A tuple (X_streams, y) of normalized data X and label y
        '''
        
        X_streams = list(X)
        idx = 0
        if ADD_VITALS:
            X_streams[idx] = tf.math.divide(tf.math.subtract(X_streams[idx], self.min_vitals), tf.math.subtract(self.max_vitals, self.min_vitals))
            idx += 1
        if ADD_LABS:
            X_streams[idx] = tf.math.divide(tf.math.subtract(X_streams[idx], self.min_labs), tf.math.subtract(self.max_labs, self.min_labs))
            idx += 1
        if ADD_NOTES:
            X_streams[idx] = tf.math.divide(tf.math.subtract(X_streams[idx], self.min_notes), tf.math.subtract(self.max_notes, self.min_notes))
        return tuple(X_streams), y


    def build_pipeline(self, icustays, output_signature, n_labels=2, batch_size=64, oversample=False, weighted=False, shuffle=True):
        '''Builds a data pipeline.

        PARAMETERS
            icustays (np.array):                Array containing icustay-ids and labels

            output_signature:                   tf-signature of the model in- and outputs

            n_labels (int):                     Number of bins used for oversampling and weighting (default: 2)

            batch_size (int):                   Batch size to be used for data 

            oversample (bool):                  Randomly oversamples data if True (default: False)

            weighted (bool):                    Generates sample weights if True (default: False)


        RETURNS
            tf-dataset containing the icustays
        '''
        # Calculate class imbalance:
        if weighted:
            # Init weights:
            r = np.empty(n_labels, dtype=float)
            
            # Fill r and print used weights:
            print("\nSample weights per window:")
            for t0, t1, i in windowed_range(n_labels):
                r[i] = icustays.shape[0] / np.sum((icustays[:,1] >= t0) & (icustays[:,1] <= t1))
                print(f"{t0:.2f} < y < {t1:.2f}: {r[i]:.2f}")
            print()
        
        elif oversample:
            # Oversample minority-class:
            ros = RandomOverSampler(random_state=self.random_state)
            l = np.minimum(n_labels * icustays[:,1], n_labels-1).astype(int) 
            icustays,_ = ros.fit_resample(icustays, l)
            
            # Print assumed labels and their counts:
            print("\nSample counts per window:")
            for t0, t1, i in windowed_range(n_labels):
                print(f"{t0:.2f} < y < {t1:.2f}: {(l==i).sum():d}")
            print()
        
        # Create datasets:
        data = tf.data.Dataset.from_generator(
            self.__icustays2data,
            args=[icustays],
            output_signature=output_signature
        ).cache()
        
        # Normalize data:
        data = data.map(self.__normalize)
        
        # Add backpropagation weight to each sample:
        if weighted:
            def map_with_weights(X, y):
                weight = tf.gather(r, tf.math.minimum(tf.cast(n_labels * y, dtype=tf.int64), n_labels-1))
                return (X, y, weight)

            data = data.map(map_with_weights)
        
        # Shuffle and batch data for each epoch:
        if shuffle:
            data = data.shuffle(5)
        data = data.padded_batch(int(batch_size), padding_values=tf.cast(-2., dtype=tf.float64))

        # Create prefetchable tf-dataset:
        data = data.prefetch(buffer_size=tf.data.AUTOTUNE)

        return data


#--------------------------------------------------------------------------------#
# Trainer classes                                                                #
#--------------------------------------------------------------------------------#

class TrainerBase:
    def __init__(self, loss, metrics, output_signature, min_los_icu, n_folds=5, n_clients=1, n_steps=100, es_metric='loss', es_mode='min', es_delta=0., es_patience=30, max_threads=1, random_state=None, i_train=None, i_test=None):
        '''Creates a new TrainerBase-object.

        PARAMETERS
            loss (tf.keras.losses):             Loss function for training

            metrics (tf.keras.metrics):         List of metrics used during training

            output_signature:                   tf-signature of the model in- and outputs
            
            min_los_icu (int):                  Minimum length of stay in icu in hours (for logging purposes)

            n_folds (int):                      Number of cross validation folds (default: 5)

            n_clients (int):                    Number of local models (default: 1)

            n_steps (int):                      Number of steps (default: 100)

            es_metric (string):                 Metric which is monitored for early stopping (default: 'loss')

            es_mode ['min', 'max']:             Whether minimal or maximal value is considered optimal for early stopping (default: 'min')

            es_delta (float):                   Minimal delta for early stopping improvements (default:0)

            es_patience (int):                  Patience parameter for early stopping in FL-rounds (default: 30)

            max_threads (int):                  Maximum number of parrallel threads (default: 1)
            
            random_state (int):                 Seed for the random generator (default: None)

        '''
        # Metrics:
        self.loss = loss
        self.metrics = metrics

        self.metric_names = ['loss']
        self.metric_names.extend([m.name for m in self.metrics])

        # Early stopping:
        self.es_metric = es_metric
        self.es_mode = es_mode
        self.es_delta = abs(es_delta)
        self.es_patience = es_patience

        # Multithreading:
        self.max_threads = max_threads
        self.lock = Lock()

        # Other:
        self.n_clients = n_clients
        self.n_folds = n_folds
        self.n_steps = n_steps

        self.batch_size = int(512/n_clients)
        self.output_signature = output_signature
        self.random_state = random_state
        self.min_los_icu = min_los_icu
        self.split_log = "./splits.json"
        
        # Store external i_train and i_test if provided
        self.i_train_external = i_train
        self.i_test_external = i_test

    def _split_test(self, n_labels=2, shuffle=False):

        # Check if external i_train and i_test are provided
        if self.i_train_external is not None and self.i_test_external is not None:
            for fold in range(1, self.n_folds + 1):
                yield self.i_train_external, self.i_test_external, fold
        else:
        
            y = np.zeros_like(self.icustays[:,1], dtype=int)
            for start, stop, i in windowed_range(n_labels):
                y[np.where(np.logical_and(self.icustays[:,1]>start, self.icustays[:,1]<=stop))] = i
    
            # For each cross-validation-fold:
            fold = 0
            for i_train, i_test in StratifiedKFold(n_splits=self.n_folds, shuffle=shuffle, random_state=self.random_state).split(self.icustays[:,0], y):
                fold += 1
    
                yield i_train, i_test, fold

    def _split_valid(self, indices, n_labels=2, shuffle=False, stratify=False):
        y = np.zeros_like(self.icustays[:,1], dtype=int)
        for start, stop, i in windowed_range(n_labels):
            y[np.where(np.logical_and(self.icustays[:,1]>start, self.icustays[:,1]<=stop))] = i

        # For each FL-client:
        client = 0
        for split in stratified_split(indices, self.n_clients, stratify=y[indices] if stratify else None, shuffle=shuffle, random_state=self.random_state):
            client += 1
            
            # Split validation data:
            i_train, i_valid = train_test_split(
                split,
                train_size=0.8,
                stratify=y[split] if stratify else None,
                shuffle=shuffle,
                random_state=self.random_state
            )

            yield i_train, i_valid, client

    def _fit_model(self, model, data_train, data_valid=None, epochs=1, callbacks=None, client=1, save_weights=True):
        # Fit model:
        history = model.fit(
            data_train,
            validation_data=data_valid,
            callbacks=callbacks,
            epochs=epochs,
            verbose=1 if self.printing_active else 0
        )

        self.lock.acquire(True)

        # Save model parameters:
        if save_weights:
            self.client_weights[client] = self._get_model_weights(model)

        self.lock.release()

        return history

    def _evaluate_model(self, model, data, client=1, fold=1, save_predictions=True):
        # Evaluate model:
        scores = model.evaluate(
            data,
            verbose=1 if self.printing_active else 0
        )

        # Create label-prediction pairs:
        preds = None
        if save_predictions:
            preds = np.full_like(self.predictions[fold-1, client-1], np.NaN, dtype=float)
            i = 0
            for X, y in data:
                n = len(y)
                preds[i:i+n, 0] = y.numpy().reshape(n)
                preds[i:i+n, 1] = model.predict(X).reshape(n)
                i += n

        self.lock.acquire(True)

        # Save label-prediction pairs:
        if save_predictions:
            self.predictions[fold-1, client-1, :, :] = preds

        # Log scores:
        if not self.printing_active:
            s = 'Scores: '
            for i in range(len(self.metric_names)):
                s += f'{self.metric_names[i]:s} = {scores[i]:.4f}; '
            self._enqueue_console(client, s)

        self.lock.release()

        del preds
        return scores

    def _add_thread(self, target, args, kwargs):
        if self.threads != None:
            # Run in parrallel threads:
            t = Thread(target=target, args=args, kwargs=kwargs)
            self.threads.append(t)

        else:
            # Print output:
            self._flush_console()

            # Run in main thread:
            target(*args, **kwargs)

            # Print output:
            self._flush_console()

    def _run_threads(self):
        if self.threads != None:
            self.printing_active = False

            active = []
            # Start threads:
            while len(active) <= self.max_threads and len(self.threads) > 0:
                t = self.threads.pop()
                active.append(t)
                t.start()

            # Wait for threads to finish:
            while len(active) > 0:
                t = active.pop(0)
                t.join()
                del t

                if len(self.threads) > 0:
                    t = self.threads.pop()
                    active.append(t)
                    t.start()

            # Clear threads:  
            self.threads.clear()

            self.printing_active = True

            # Print output:
            self._flush_console()

    def _set_model_weights(self, model, weights):
        for i in range(len(weights)):
            model.layers[i].set_weights(weights[i])

    def _get_model_weights(self, model):
        return [layer.get_weights() for layer in model.layers]

    def _init_model_weights(self, model, path='./data/default_weights.h5'):
        # If default weights were previously stored, load them:
        if exists(path):
            model.load_weights(path)

        # If not, save the current model weights as default:
        else:
            model.save_weights(path)

        # Finally update the global weights:
        self.global_weights = self._get_model_weights(model)

    def _enqueue_console(self, client, text):
        self.console_buffer[client-1] += text + '\n'

    def _flush_console(self):
        for i in range(len(self.console_buffer)):
            print(self.console_buffer[i], end='')
            self.console_buffer[i] = ''

    def reset(self):
        self.lock.acquire(True)

        # Results:
        self.train_scores = {l: np.full((self.n_folds, self.n_clients, self.n_steps), np.NaN, dtype=float) for l in self.metric_names}
        self.valid_scores = {l: np.full((self.n_folds, self.n_clients, self.n_steps), np.NaN, dtype=float) for l in self.metric_names}
        self.test_scores = {l: np.zeros(self.n_folds, dtype=float) for l in self.metric_names}
        self.predictions = np.full((self.n_folds, self.n_clients, int(np.ceil(float(len(self.icustays)) / float(self.n_folds))), 2), np.NaN, dtype=float)

        # Console output:
        self.console_buffer = ['' for i in range(self.n_clients)]
        self.printing_active = True

        # Multithreading:
        self.threads = [] if self.max_threads > 1 and self.n_clients > 1 else None

        # Weights:
        self.global_weights = None
        self.client_weights = {}

        self.lock.release()

    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump(
                (self.train_scores, self.valid_scores, self.test_scores, self.predictions),
                file
            )

    def load(self, path):
        try:
            with open(path, 'rb') as file:
                self.train_scores, self.valid_scores, self.test_scores, self.predictions = pickle.load(
                    file
                )
        except:
            with open(path, 'rb') as file:
                self.train_scores, self.valid_scores, self.test_scores = pickle.load(
                    file
                )

    def plot_history(self, key, ax, x_step=2, client=None):
        '''Prints a specific metric from a list of tf.history objects.

        PARAMETERS
            key (string):       Name of the metric to be sampled

            ax (plt.axes):      Pyplot axes object which should be used for plotting

            x_step (int):       Label step of x-values

            client (any):       Dictionary key of the client to be sampled (default: None)
        '''

        # Get list of dictionaries:
        values_train = (self.train_scores[key][:, client-1] if client != None else self.train_scores[key][:, :])
        while len(values_train.shape) > 1:
            values_train = np.nanmean(values_train, axis=0) 
        n_train = values_train.shape[0]

        values_valid = (self.valid_scores[key][:, client-1] if client != None else self.valid_scores[key][:, :])
        while len(values_valid.shape) > 1:
            values_valid = np.nanmean(values_valid, axis=0)
        n_valid = values_valid.shape[0]

        ax.plot(np.arange(1,n_train+1), values_train, label='train')
        ax.plot(np.arange(1,n_valid+1), values_valid, label='valid')

        ax.set_xticks(np.arange(max(n_valid, n_train), step=x_step))
        ax.set_xlabel('epoch')
        ax.set_title(key)
        ax.legend()

    def set_data(self, vitals, labs=None, notes=None, key='icustay_id', label='label_death_icu'):
        '''Sets the data to be used by the trainer.

        PARAMETERS
            vitals (Pandas DataFrame):  Vital values grouped by ICU-stay_id

            labs (Pandas DataFrame):    Lab values grouped by ICU-stay_id

            notes (Pandas DataFrame):   Notes values grouped by ICU-stay_id

            key (string):               Name of the column to be used as key (default: 'icustay_id')

            label (string):             Name of the column to be used as label (default: 'label_death_icu')
        '''
        print(f'Importing data...')

        # Set up list of key-label-pairs:
        self.icustays = vitals[[key, label]].groupby([key]).first().reset_index().to_numpy()

        # Group vitals by key:
        self.vitals = vitals.drop(label, axis=1).groupby([key]) if ADD_VITALS else None
        # Group labs by key:
        self.labs = labs.drop(label, axis=1).groupby([key]) if ADD_LABS else None
        # Group notes by key:
        self.notes = notes.drop(label, axis=1).groupby([key]) if ADD_NOTES else None

        # Save key:
        self.key = key

        # Reset results:
        self.reset()

        print(f'Done. Imported {len(self.icustays):d} patients.')
        print(f'key:    {key:s}')
        print(f'label:  {label:s}')
        if ADD_VITALS:
            print(f'vitals: {", ".join(self.vitals.get_group(self.icustays[0,0]).drop(key, axis=1).columns):s}')
        if ADD_LABS:
            print(f'labs:   {", ".join(self.labs.get_group(self.icustays[0,0]).drop(key, axis=1).columns):s}')
        if ADD_NOTES:
            print(f'notes:  {", ".join(self.notes.get_group(self.icustays[0,0]).drop(key, axis=1).columns):s}')

class Trainer(TrainerBase):
    def __init__(self, loss, metrics, output_signature, min_los_icu, n_folds=5, n_clients=1, n_epochs=100, es_metric='loss', es_mode='min', es_delta=0., es_patience=30, max_threads=1, random_state=None, i_train=None, i_test=None):
        '''Creates a new Trainer-object.

        PARAMETERS
            loss (tf.keras.losses):             Loss function for training

            metrics (tf.keras.metrics):         List of metrics used during training

            output_signature:                   tf-signature of the model in- and outputs
            
            min_los_icu:                        Minimum length of stay in icu in hours (for logging purposes)

            es_metric:                          Metric which is monitored for early stopping (default: 'loss')

            es_mode ['min', 'max']:             Whether minimal or maximal value is considered optimal for early stopping (default: 'min')

            es_delta (float):                   Minimal delta for early stopping improvements (default:0)

            es_patience (int):                  Patience parameter for early stopping in FL-rounds (default: 30)

            max_threads (int):                  Maximum number of parrallel threads (default: 1)
            
            random_state (int):                 Seed for the random generator (default: None)

        '''
        super().__init__(
            loss, metrics, output_signature, min_los_icu, 
            n_folds=n_folds, n_clients=n_clients, n_steps=n_epochs,
            es_metric=es_metric, es_mode=es_mode, es_delta=es_delta, es_patience=es_patience,
            max_threads=max_threads,
            random_state=random_state,
            i_train=i_train,  # Pass i_train as an argument to TrainerBase
            i_test=i_test    # Pass i_test as an argument to TrainerBase       
        )

    def __train_eval_model(self, model, client=1, fold=1, epochs=1, callbacks=None, data_train=None, data_valid=None, data_test=None):
        # Fit model:
        history = self._fit_model(
            model,
            data_train,
            data_valid=data_valid,
            callbacks=callbacks,
            epochs=epochs,
            client=client
        )

        # Evaluate model:
        if self.printing_active:
            print('\nEvaluation:')

        scores = self._evaluate_model(
            model,
            data_test,
            client=client,
            fold=fold
        )

        # Save results:
        self.lock.acquire(True)

        for i in range(len(self.metric_names)):
            l = self.metric_names[i]

            self.train_scores[l][fold-1, client-1, :len(history.history[l])] = history.history[l]
            self.valid_scores[l][fold-1, client-1, :len(history.history['val_'+l])] = history.history['val_'+l]
            self.test_scores[l][fold-1] += scores[i]

        self.lock.release()

        del history
        del scores
        del model

    def evaluate(self, model, n_labels=2, shuffle=False, oversample=False, weighted=False, stratify_clients=False):
        '''Evaluates a model.

        PARAMETERS
            model (tf.keras.model):     Model to evaluate

            n_labels (int):             Number of bins used for oversampling and weighting (default: 2)

            shuffle (bool):             Determines whether data is shuffeled before creating splits (default: False)

            oversample (bool):          Randomly oversamples data if True (default: False)

            weighted (bool):            Generates sample weights if True (default: False)

            stratify_clients (bool):    Generates stratified client splits if True (default: False)
        '''

        # Init properties:
        self.reset()

        # Init log file for batches:
        if self.n_clients > 1:
            self.split_log = f'./data/min{self.min_los_icu:d}h/splits_lml{self.n_clients:d}.json'
        else:
            self.split_log = f'./data/min{self.min_los_icu:d}h/splits_cml.json'
        
        with open(self.split_log, 'wt') as log:
            log.write( '{\n')
            log.write(f'  "min_los_icu":{self.min_los_icu:d},\n')
            log.write(f'  "n_folds":{self.n_folds:d},\n')
            log.write(f'  "n_clients":{self.n_clients:d},\n')
            log.write( '  "folds":[\n')

        # For each cross-validation-fold:
        for i_rest, i_test, fold in self._split_test(n_labels=n_labels, shuffle=shuffle):

            # Log fold:
            with open(self.split_log, 'at') as log:
                log.write( '    {\n')
                log.write(f'      "fold":{fold:d},\n')
                log.write(f'      "ids_test":[{",".join(str(id) for id in self.icustays[i_test,0]):s}],\n')
                log.write( '      "clients":[\n')

            # For each client:
            for i_train, i_valid, client in self._split_valid(i_rest, n_labels=n_labels, shuffle=shuffle, stratify=stratify_clients):

                # Print header:
                self._enqueue_console(client,
                    f'\n---------------------------------------------------------------------------' +
                    f'\nCross-validation iteration {fold:d}/{self.n_folds:d}; Client {client:d}/{self.n_clients:d}' +
                    f'\nTraining size = {len(i_train):d}; Validation size = {len(i_valid):d}; Test size = {len(i_test):d}' +
                    f'\nBatch size = {self.batch_size:d}' +
                    f'\n---------------------------------------------------------------------------'
                )

                # Log patients:
                with open(self.split_log, 'at') as log:
                    log.write( '        {\n')
                    log.write(f'          "client":{client:d},\n')
                    log.write(f'          "ids_train":[{",".join(str(id) for id in self.icustays[i_train,0]):s}],\n')
                    log.write(f'          "ids_valid":[{",".join(str(id) for id in self.icustays[i_valid,0]):s}]\n')
                    log.write( '        }' +  (',\n' if client < self.n_clients else '\n'))

                # Copy model:
                local_model = tf.keras.models.clone_model(model)

                if MODEL_ARCHITECTURE == "transformer":
                    optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=lr_schedule, weight_decay=weight_decay, clipnorm=clip_norm)
                else:
                    optimizer = tf.keras.optimizers.Adam(0.01)
                    
                # Compile model:
                local_model.compile(
                    loss=self.loss,
                    optimizer=optimizer,
                    metrics=self.metrics
                )

                # Load starting weights:
                self._init_model_weights(local_model)

                # Build train- and validation pipelines:
                with redirect_stdout(io.StringIO()) as out:
                    builder = DataBuilder(
                        self.icustays[np.concatenate((i_train, i_valid), axis=None)],
                        self.vitals, self.labs, self.notes,
                        key=self.key, random_state=self.random_state
                    )
                    data_train = builder.build_pipeline(self.icustays[i_train], self.output_signature, batch_size=self.batch_size, n_labels=n_labels, oversample=oversample, weighted=weighted)
                    data_valid = builder.build_pipeline(self.icustays[i_valid], self.output_signature, batch_size=self.batch_size, n_labels=n_labels)
                    data_test =  builder.build_pipeline(self.icustays[i_test],  self.output_signature, batch_size=self.batch_size, n_labels=n_labels, shuffle=False)
                self._enqueue_console(client, '\n' + out.getvalue())
                del out

                # Callbacks:
                callbacks=[
                    tf.keras.callbacks.LearningRateScheduler(lambda epoch, eta: 0.5*eta if (epoch%5) == 0 and epoch > 0 else eta),
                    tf.keras.callbacks.EarlyStopping(
                        patience=self.es_patience,
                        monitor='val_'+self.es_metric,
                        mode=self.es_mode,
                        min_delta=self.es_delta,
                        restore_best_weights=True
                    )
                ]

                # Train each client in its own thread:
                self._add_thread(
                    target=self.__train_eval_model, 
                    args=(local_model,),
                    kwargs={
                        'client': client,
                        'fold': fold,
                        'epochs': self.n_steps,
                        'callbacks': callbacks, 
                        'data_train': data_train,
                        'data_valid': data_valid,
                        'data_test': data_test
                    }
                )
            
            # Close fold in log:
            with open(self.split_log, 'at') as log:
                log.write( '      ]\n')
                log.write( '    }' +  (',\n' if fold < self.n_folds else '\n'))

            # Run all threads:
            self._run_threads()

        # Close object in log:
        with open(self.split_log, 'at') as log:
            log.write( '  ]\n')
            log.write( '}\n')

        for key in self.test_scores:
            self.test_scores[key] /= self.n_clients

class TrainerFL(TrainerBase):
    def __init__(self, loss, metrics, output_signature, min_los_icu, n_folds=5, n_clients=1, n_rounds=100, n_epochs=1, es_metric='loss', es_mode='min', es_delta=0., es_patience=30, max_threads=1, random_state=None, i_train=None, i_test=None):
        '''Creates a new TrainerFL-object.

        PARAMETERS
            loss (tf.keras.losses):             Loss function for training

            metrics (tf.keras.metrics):         List of metrics used during training

            output_signature:                   tf-signature of the model in- and outputs
            
            min_los_icu:                        Minimum length of stay in icu in hours (for logging purposes)

            n_folds (int):                      Number of cross validation folds (default: 5)

            n_clients (int):                    Number of local models (default: 1)

            n_rounds (int):                     Number of FL-rounds (default: 100)

            n_epochs (int):                     Number of local epochs per FL-round (default: 1)

            es_metric:                          Metric which is monitored for early stopping (default: 'loss')

            es_mode ['min', 'max']:             Whether minimal or maximal value is considered optimal for early stopping (default: 'min')

            es_delta (float):                   Minimal delta for early stopping improvements (default:0)

            es_patience (int):                  Patience parameter for early stopping in FL-rounds (default: 30)

            max_threads (int):                  Maximum number of parrallel threads (default: 1)
            
            random_state (int):                 Seed for the random generator (default: None)

        '''
        super().__init__(
            loss, metrics, output_signature, min_los_icu, 
            n_folds=n_folds, n_clients=n_clients, n_steps=n_rounds,
            es_metric=es_metric, es_mode=es_mode, es_delta=es_delta, es_patience=es_patience,
            max_threads=max_threads,
            random_state=random_state,
            i_train=i_train,  # Pass i_train as an argument to TrainerBase
            i_test=i_test    # Pass i_test as an argument to TrainerBase
        )
        self.n_epochs=n_epochs
        
        # Initialize a list to store client data
        self.client_data = []

        # Initialize a list to store test data
        self.test_data = []

    def __train_model(self, model, client=1, fold=1, fl_round=-1, epochs=1, callbacks=None, data_train=None):
        # Fit model:
        history = self._fit_model(
            model,
            data_train,
            data_valid=None,
            callbacks=callbacks,
            epochs=epochs,
            client=client
        )

        # Save results:
        self.lock.acquire(True)

        for i in range(len(self.metric_names)):
            l = self.metric_names[i]
            self.train_scores[l][fold-1, client-1, fl_round] = history.history[l][-1]

        self.lock.release()

        del history

    def __eval_model(self, model, client=1, fold=1, fl_round=-1, data_valid=None):
        # Evaluate model:
        scores = self._evaluate_model(
            model,
            data_valid,
            client=client,
            fold=fold,
            save_predictions=False
        )

        # Save results:
        self.lock.acquire(True)

        for i in range(len(self.metric_names)):
            l = self.metric_names[i]
            self.valid_scores[l][fold-1, client-1, fl_round] = scores[i]

        self.lock.release()

        del scores
        del model

    def evaluate(self, model, n_labels=2, shuffle=False, oversample=False, weighted=False, stratify_clients=False):
        '''Evaluates a model with federated learning.

        PARAMETERS
            model (tf.keras.model):     Model to evaluate

            n_labels (int):             Number of bins used for oversampling and weighting (default: 2)

            shuffle (bool):             Determines whether data is shuffeled before creating splits (default: False)

            oversample (bool):          Randomly oversamples data if True (default: False)

            weighted (bool):            Generates sample weights if True (default: False)

            stratify_clients (bool):    Generates stratified client splits if True (default: False)
        '''

        # Reset properties:
        self.reset()

        # Init log file for batches:
        self.split_log = f'./data/min{self.min_los_icu:d}h/splits_fl{self.n_clients:d}.json'
        with open(self.split_log, 'wt') as log:
            log.write( '{\n')
            log.write(f'  "min_los_icu":{self.min_los_icu:d},\n')
            log.write(f'  "n_folds":{self.n_folds:d},\n')
            log.write(f'  "n_clients":{self.n_clients:d},\n')
            log.write( '  "folds":[\n')

        # For each cross-validation-fold:
        for i_rest, i_test, fold in self._split_test(n_labels=n_labels, shuffle=shuffle):

            # Set i_train
            #i_rest = self.i_train
            # Set i_test
            #i_test = self.i_test
            
            # New normalization bounds:
            builder = DataBuilder(
                self.icustays[i_rest], 
                self.vitals, self.labs, self.notes, 
                key=self.key, random_state=self.random_state
            )

            # Log fold:
            log_buffer  =  '    {\n'
            log_buffer += f'      "fold":{fold:d},\n'
            log_buffer += f'      "ids_test":[{",".join(str(id) for id in self.icustays[i_test,0]):s}],\n'
            log_buffer +=  '      "clients":[\n'

            # Build client data splits:
            clients = {}
            
            # Create test data once
            test_data = builder.build_pipeline(self.icustays[i_test], self.output_signature, batch_size=self.batch_size, n_labels=n_labels, shuffle=False)  

            for i_train, i_valid, client in self._split_valid(i_rest, n_labels=n_labels, shuffle=shuffle, stratify=stratify_clients):
                # Print header:
                print(
                    f'\n---------------------------------------------------------------------------' +
                    f'\nCross-validation iteration {fold:d}/{self.n_folds:d}; Client {client:d}/{self.n_clients:d}' +
                    f'\nTraining size = {len(i_train):d}; Validation size = {len(i_valid):d}' +
                    f'\nBatch size = {self.batch_size:d}' +
                    f'\n---------------------------------------------------------------------------'
                )

                # Log patients:
                log_buffer +=  '        {\n'
                log_buffer += f'          "client":{client:d},\n'
                log_buffer += f'          "ids_train":[{",".join(str(id) for id in self.icustays[i_train,0]):s}],\n'
                log_buffer += f'          "ids_valid":[{",".join(str(id) for id in self.icustays[i_valid,0]):s}]\n'
                log_buffer +=  '        }' +  (',\n' if client < self.n_clients else '\n')

                # Create datasets and model:
                clients[client] = {
                    'model':        tf.keras.models.clone_model(model),
                    'data_train':   builder.build_pipeline(self.icustays[i_train], self.output_signature, batch_size=self.batch_size, n_labels=n_labels, oversample=oversample, weighted=weighted), 
                    'data_valid':   builder.build_pipeline(self.icustays[i_valid], self.output_signature, batch_size=self.batch_size, n_labels=n_labels),
                    'data_test':    test_data,
                    'n':            len(i_train)
                }

                if MODEL_ARCHITECTURE == "transformer":
                    optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=lr_schedule, weight_decay=weight_decay, clipnorm=clip_norm)
                else:
                    optimizer = tf.keras.optimizers.Adam(0.01)
                
                # Compile model:
                clients[client]['model'].compile(
                    loss=self.loss,
                    optimizer=optimizer,
                    metrics=self.metrics
                )

                # Init model weights:
                self._init_model_weights(clients[client]['model'])

            self.client_data.append(clients)  # Store the client data for each fold

            # Close fold in log:
            with open(self.split_log, 'at') as log:
                log.write( log_buffer)
                log.write( '      ]\n')
                log.write( '    }' +  (',\n' if fold < self.n_folds else '\n'))
            del log_buffer

            # For each FL-round:
            best_es = (-np.Inf if self.es_mode == 'max' else np.Inf, -1, None)
            for round in range(self.n_steps):
                # Print header:
                print(
                    f'\n---------------------------------------------------------------------------' +
                    f'\nCross-validation iteration {fold:d}/{self.n_folds:d}; Round {round+1:d}/{self.n_steps:d}' +
                    f'\n---------------------------------------------------------------------------'
                )

                # Train models:
                for client in clients:
                    # Print header:
                    if self.threads == None:
                        print(f'\nTraining client {client:d}/{self.n_clients:d}:')

                    # Callbacks:
                    callbacks=[
                        tf.keras.callbacks.LearningRateScheduler(lambda epoch, eta: 0.5*eta if epoch == 0 and (round%5) == 0 and round > 0 else eta)
                    ]

                    # Train each client in its own thread:
                    self._add_thread(
                        target=self.__train_model,
                        args=(clients[client]['model'],),
                        kwargs={
                            'client': client,
                            'fold': fold,
                            'fl_round': round,
                            'epochs': self.n_epochs,
                            'callbacks': callbacks, 
                            'data_train': clients[client]['data_train']
                        }
                    )

                # Run all threads:
                self._run_threads()

                # Calculate average weights:
                self.global_weights = []
                n = np.sum([clients[client]['n'] for client in clients], dtype=float)
                for client in self.client_weights:
                    frac = float(clients[client]['n']) / n
                    print(f'Factor client {client:d}: {frac:.2f}')
                    for i in range(len(self.client_weights[client])):
                        if len(self.global_weights) <= i:
                            self.global_weights.append([frac * self.client_weights[client][i][j] for j in range(len(self.client_weights[client][i]))])
                        else:
                            self.global_weights[i] = [self.global_weights[i][j] + (frac * self.client_weights[client][i][j]) for j in range(len(self.global_weights[i]))]

                # Validate models:
                for client in clients:
                    # Set model weights:
                    self._set_model_weights(clients[client]['model'], self.global_weights)

                    # Print header:
                    self._enqueue_console(client, f'\nValidation client {client:d}/{self.n_clients:d}:')

                    # Validate each client in its own thread:
                    self._add_thread(
                        target=self.__eval_model,
                        args=(clients[client]['model'],),
                        kwargs={
                            'client': client,
                            'fold': fold,
                            'fl_round': round,
                            'data_valid': clients[client]['data_valid']
                        }
                    )

                # Run all threads:
                self._run_threads()

                # Early stopping:
                val_es = self.valid_scores[self.es_metric][fold-1, :, round].mean()
                if ((self.es_mode == 'max') and (val_es > best_es[0] + self.es_delta)) or ((self.es_mode == 'min') and (val_es < best_es[0] - self.es_delta)):
                    best_es = (val_es, round, copy.deepcopy(self.global_weights))
                    print(f'\nEarly stopping [round {round+1:d}]: Best {self.es_metric:s} {val_es:.2f} stored for {len(self.global_weights):d} layers')

                elif (round - best_es[1]) >= self.es_patience:
                    print(f'\nEarly stopping [round {round+1:d}]: Stopping training (Best round: {best_es[1]+1:d})')
                    break

            print(
                f'\n---------------------------------------------------------------------------' +
                f'\nCross-validation iteration {fold:d}/{self.n_folds:d}; Global Model' +
                f'\nTest size = {len(i_test):d}' +
                f'\nBatch size = {self.batch_size:d}' +
                f'\n---------------------------------------------------------------------------'
            )

            # Copy model:
            global_model = tf.keras.models.clone_model(model)

            # Compile global model:
            global_model.compile(
                loss=self.loss,
                metrics=self.metrics
            )

            # Set global model:
            self._set_model_weights(global_model, best_es[2])

            # Evaluate global model:
            data_test = builder.build_pipeline(self.icustays[i_test], self.output_signature, batch_size=self.batch_size, n_labels=n_labels)
            scores = self._evaluate_model(global_model, data_test, fold=fold)

            # Save scores:
            for i in range(len(self.metric_names)):
                self.test_scores[self.metric_names[i]][fold-1] = scores[i]

            #clear memory:
            del data_test
            del clients
            del scores
            del global_model

        # Close object in log:
        with open(self.split_log, 'at') as log:
            log.write( '  ]\n')
            log.write( '}\n')