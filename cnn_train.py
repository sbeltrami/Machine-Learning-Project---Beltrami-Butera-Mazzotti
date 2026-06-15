#Import the libraries
import numpy as np
import xarray as xr
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import ParameterGrid
import dill
import gc
import os
import json
import logging
import multiprocessing as mp
from datetime import datetime

# Implement a Convolutional Neural Network (CNN) to predict the Dipole Index during September-November (SON), by using the architecture proposed by Tao (2024) DOI 10.1088/1748-9326/ad7522

# Define the months for the CNN input; define the number of lead time
months_name = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG']
number_lead_time = 6

# Define the values of all the possible hyperparameters of the CNN
grid_search_params = {
    'filter_size': [(3,3), (5,5)],
    'initial_filters': [8, 16, 32],
    'initial_dropout': [0, 0.1, 0.2, 0.3],
    'dense_units': [50, 100, 150],
    'batch_size': [8, 16, 32, 64],
    'learning_rate': [0.1, 0.01, 0.001]
}
# Generate all the possible combinations for the hyperparameters (864)
param_grid = list(ParameterGrid(grid_search_params))


# Logging pre-process
def get_logger(lead_time):
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger(f"LT_{lead_time}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(f"logs/leadtime_{lead_time}.log")
        sh = logging.StreamHandler()
        fmt = logging.Formatter(f"%(asctime)s [LT={lead_time}] %(message)s")
        fh.setFormatter(fmt); sh.setFormatter(fmt)
        logger.addHandler(fh); logger.addHandler(sh)
    return logger

# Checkpoint
def checkpoint_path(j):
    return f"checkpoints/leadtime_{j}.json"

def load_checkpoint(j):
    path = checkpoint_path(j)
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        weights_path = f"checkpoints/leadtime_{j}_weights.npy"
        weights = list(np.load(weights_path, allow_pickle=True)) if os.path.exists(weights_path) else []
        return data, weights
    return None, []

def save_checkpoint(j, params_done, top_scores, top_models_params,
                    top_histories, top_models_weights):
    os.makedirs("checkpoints", exist_ok=True)
    data = {
        "params_done": params_done,
        "top_scores": top_scores,
        "top_models_params": top_models_params,
        "top_histories": top_histories,
    }
    np.save(f"checkpoints/leadtime_{j}_weights.npy",
            np.array(top_models_weights, dtype=object), allow_pickle=True)
    with open(checkpoint_path(j), "w") as f:
        json.dump(data, f)

# Function for the training of each lead time; each lead time is parallelized
def train_lead_time(j):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    log = get_logger(j)
    log.info(f"Process started (PID={os.getpid()})")

    # Upload the data in the file pickle
    log.info("Loading data")
    with open('cnn_obs.pkl', 'rb') as file:
        hadisst_dic = dill.load(file)
        ncep_ncar_dic = dill.load(file)

    # Training and validation data
    X_train = xr.concat([
        hadisst_dic[f'{months_name[-j]} (Indian Pacific oceans)']['standardized anomaly training'],
        hadisst_dic[f'{months_name[-j-1]} (Indian Pacific oceans)']['standardized anomaly training'],
        hadisst_dic[f'{months_name[-j-2]} (Indian Pacific oceans)']['standardized anomaly training'],
        ncep_ncar_dic[f'{months_name[-j]} (Indian Pacific oceans)']['standardized anomaly training'],
        ncep_ncar_dic[f'{months_name[-j-1]} (Indian Pacific oceans)']['standardized anomaly training'],
        ncep_ncar_dic[f'{months_name[-j-2]} (Indian Pacific oceans)']['standardized anomaly training'],
    ], dim='channels').fillna(0).transpose('time', 'lat', 'lon', 'channels')

    y_train = hadisst_dic['SON (di west-east)']['dipole index training'].fillna(0)

    X_val = xr.concat([
        hadisst_dic[f'{months_name[-j]} (Indian Pacific oceans)']['standardized anomaly validation'],
        hadisst_dic[f'{months_name[-j-1]} (Indian Pacific oceans)']['standardized anomaly validation'],
        hadisst_dic[f'{months_name[-j-2]} (Indian Pacific oceans)']['standardized anomaly validation'],
        ncep_ncar_dic[f'{months_name[-j]} (Indian Pacific oceans)']['standardized anomaly validation'],
        ncep_ncar_dic[f'{months_name[-j-1]} (Indian Pacific oceans)']['standardized anomaly validation'],
        ncep_ncar_dic[f'{months_name[-j-2]} (Indian Pacific oceans)']['standardized anomaly validation'],
    ], dim='channels').fillna(0).transpose('time', 'lat', 'lon', 'channels')

    y_val = hadisst_dic['SON (di west-east)']['dipole index validation'].fillna(0)

    X_train_tensor = tf.constant(X_train.values.astype('float32'))
    y_train_np = y_train.values.astype('float32')
    X_val_tensor = tf.constant(X_val.values.astype('float32'))
    y_val_np = y_val.values.astype('float32')
    log.info(f"Shape input: {X_train_tensor.shape}")

    # Upload Checkpoint
    ckpt, top_models_weights = load_checkpoint(j)
    if ckpt:
        params_done = ckpt["params_done"]
        top_scores = ckpt["top_scores"]
        top_models_params = ckpt["top_models_params"]
        top_histories = ckpt["top_histories"]
        log.info(f"Checkpoint found: combination {params_done}/{len(param_grid)}")
    else:
        params_done = 0
        top_scores = []
        top_models_params = []
        top_histories = []
        top_models_weights = []

    # Grid search
    oom_count = 0
    for i, params in enumerate(param_grid):
        if i < params_done:
            continue

        log.info(f"Combination {i+1}/{len(param_grid)} | {params}")
        tf.keras.backend.clear_session()
        gc.collect()

        # Build model
        model = models.Sequential()

        filters = params['initial_filters']
        dropout = params['initial_dropout']

        model.add(layers.Conv2D(filters, params['filter_size'], activation='elu', input_shape=X_train_tensor.shape[1:], padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D())
        model.add(layers.Dropout(dropout))

        filters *= 2
        dropout += 0.1
        model.add(layers.Conv2D(filters, params['filter_size'], activation='elu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D())
        model.add(layers.Dropout(dropout))

        filters *= 2
        dropout += 0.1
        model.add(layers.Conv2D(filters, params['filter_size'], activation='elu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D())
        model.add(layers.Dropout(dropout))

        model.add(layers.Flatten())
        model.add(layers.Dense(params['dense_units'], activation='elu'))
        model.add(layers.Dense(1))

        model.compile(
            optimizer=optimizers.Adam(learning_rate=params['learning_rate']),
            loss='mse', metrics=['mae']
        )

        lr_scheduler = callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=10, verbose=0)
        early_stop = callbacks.EarlyStopping(
            patience=100, restore_best_weights=True)

        try:
            history = model.fit(
                X_train_tensor, y_train_np,
                validation_data=(X_val_tensor, y_val_np),
                epochs=100,
                batch_size=params['batch_size'],
                callbacks=[lr_scheduler, early_stop],
                verbose=0
            )
        except tf.errors.ResourceExhaustedError:
            oom_count += 1
            log.warning(f"OOM skipped #{oom_count}: {params}")
            del model
            tf.keras.backend.clear_session()
            gc.collect()
            params_done = i + 1
            save_checkpoint(j, params_done, top_scores, top_models_params,
                            top_histories, top_models_weights)
            continue

        val_loss = min(history.history['val_loss'])
        log.info(f"val_loss={val_loss:.6f}")

        if len(top_scores) < 10:
            top_models_weights.append(model.get_weights())
            top_models_params.append(params)
            top_scores.append(val_loss)
            top_histories.append(history.history)
        else:
            idx = np.argmax(top_scores)
            if val_loss < top_scores[idx]:
                top_models_weights[idx] = model.get_weights()
                top_models_params[idx] = params
                top_scores[idx] = val_loss
                top_histories[idx] = history.history

        del model
        tf.keras.backend.clear_session()
        gc.collect()

        params_done = i + 1
        if params_done % 10 == 0:
            save_checkpoint(j, params_done, top_scores, top_models_params, top_histories, top_models_weights)
            log.info(f"Saved checkpoint ({params_done}/{len(param_grid)})")

    # Final Checkpoint
    save_checkpoint(j, len(param_grid), top_scores, top_models_params, top_histories, top_models_weights)
    log.info(f"COMPLETED. Top 3 val_loss: {sorted(top_scores)[:3]}")

    return j, top_scores, top_models_params, top_histories, top_models_weights

# Main
if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Main logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [MAIN] %(message)s",
        handlers=[
            logging.FileHandler(f"logs/main_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    log = logging.getLogger("MAIN")
    log.info(f"Starting parallel training on {number_lead_time} lead time")
    log.info(f"Combinations for lead time: {len(param_grid)}")
    log.info(f"Total combinations: {len(param_grid) * number_lead_time}")

    # Run of the six processes in parallel (one for each lead time)
    with mp.Pool(processes=number_lead_time) as pool:
        results = pool.map(train_lead_time, range(1, number_lead_time + 1))

    # Save the final results
    results_per_leadtime = {}
    for j, top_scores, top_models_params, top_histories, top_models_weights in results:
        results_per_leadtime[j] = {
            'top_scores':         top_scores,
            'top_models_params':  top_models_params,
            'top_histories':      top_histories,
            'top_models_weights': top_models_weights,
        }

    log.info("Saving the final results in results_per_leadtime.pkl...")
    # Create file pickle
    with open("results_per_leadtime.pkl", "wb") as f:
        dill.dump(results_per_leadtime, f)
    log.info("Everything completed!")