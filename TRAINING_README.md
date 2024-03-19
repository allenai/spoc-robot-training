# Data Preprocessing

Before training a model, ensure that you have generated the metadata for your dataset. You can do this by running the following command:

```bash
export PYTHONPATH="./"
python -m scripts.preprocess_dataset --dataset_lists DATASETNAMES --dataset_path /data/datasets
```

If you already have the files `DATASET_NAME/house_id_to_sub_house_id_train.json` and `DATASET_NAME/house_id_to_sub_house_id_val.json`, you can skip this step.

# Training a Model

To train a model, use the following command:

```bash
python -m training.offline.train_pl --dataset_version DATASET_NAME --eval_every 50 --precision PRECISION_TYPE --input_sensors LIST__OF_SENSORS --per_gpu_batch PER_GPU_BATCH_SIZE --sliding_window 50 --model MODEL_NAME --model_version base  --lr BASE_LEARNING_RATE
```

You can set the precision to fixed precision:

```bash
--precision 32-true
```

Or mixed precision:

```bash
--precision 16-mixed
```

Example Full Command:

```bash
python -m training.offline.train_pl --max_samples MAX_TRAIN_SAMPLES --eval_max_samples MAX_EVAL_SAMPLES --eval_every EVAL_EPOCH_FREQ --save_every SAVE_EPOCH_FREQ --model_version siglip_base_3 --sliding_window 4 --per_gpu_batch 3 --lr 0.0002 --data_dir DATASET_DIR --dataset_version CHORES --model EarlyFusionCnnTransformer --input_sensors raw_navigation_camera raw_manipulation_camera last_actions an_object_is_in_hand --precision 16-mixed --loss action --max_epochs MAX_EPOCH --resume_local --output_dir OUTPUT_DIR --num_nodes NUM_NODES
```

If you want to train a model on multiple datasets, First define your mixture in `training/offline/dataset_mixtures.py`, for example, `TWO_TASK_MIX = ['ObjectNavType', 'ObjectNavDescription']`. Then pass that name as an argument: `--dataset_version TWO_TASK_MIX`.

# Evaluating a Model (Online Evaluation)

To evaluate a model using online evaluation, use the following command:


```bash
python -m training.offline.online_eval --shuffle --eval_subset minival --output_basedir OUTPUT_DIR --test_augmentation --eval_set_size 200 --task_type TASK_TYPE  --max_eps_len 600 --input_sensors raw_navigation_camera raw_manipulation_camera last_actions an_object_is_in_hand --house_set objaverse --training_run_id TRAINING_ID --wandb_logging False --ckptStep STEP_NUMBER --num_workers NUM_WORKERS --gpu_devices LIST_OF_GPU_IDS
```

To log the metrics and model weights locally use the following tags:

```bash
--wandb_logging False --output_dir OUTPUT_DIR
```

The weights and metrics will be saved in OUTPUT_DIR. An experiment ID will be assigned to your experiment. Use the following to evaluate the trained model. 

```bash
--wandb_logging False --output_basedir OUTPUT_DIR --training_run_id TRAINING_ID --ckptStep STEP_NUMBER 
```

Note that you need to pass the same directory to both training and evaluation. 

If you would prefer to use wandb then the TRAINING_ID will be your wandb run id and you can use the following tags during training and evaluation:

```bash
--wandb_logging True --wandb_entity_name WANDB_ENTITY_NAME --wandb_project_name WANDB_PROJECT_NAME
```

Note that if you are using wandb you should also specify a local folder in which your data will be saved before being uploaded to wandb. You can do so by setting the environment variable of WANDB_DIR:

```bash
export WANDB_DIR=/local/path/to/save/files
```

# Creating Your Own Model

To create your own model, follow these steps:

1. Define a new model class in `architecture/models/transformer_models` (e.g., `early_fusion_tsfm_models.py`).

   Requirements for implementation:

   - A class describing your model.
   - A `forward(self, model_inputs, actions)` function that calculates the loss and returns a dictionary with keys "logits" and "loss."
   - A class method called `build_model` that takes "model_version," "input_sensors," "loss," and "ckpt_path" as input and returns the model and preprocessor as output.
   - A class method called `build_agent` that takes "model_version," "input_sensors," "loss," "device," "sampling" (greedy or sampling), and "ckpt_path" as input and returns the online evaluation agent as output.

2. Register your model in `architecture/models/transformer_models/__init__.py`.

3. Implement the `Agent` class that takes care of caching inputs during online evaluation and performs the forward pass during online evaluation.

# Training and evaluating an online RL baseline

First, export the Obajverse data directory and the houses directory:

```bash
export OBJAVERSE_DATA_DIR=/path/to/objaverse_assets
export OBJAVERSE_HOUSES_DIR=/path/to/objaverse_houses
```

To train a model, use the following command:

```bash
python -m allenact.main training/online/siglip_vitb_gru_rgb_augment_objectnav
```

The results will be saved to `experiment_output`. Please refer to [AllenAct](https://allenact.org/) for more details.

You can use wandb callbacks with the `WANDB_PROJECT` and `WANDB_ENTITY` specified:

```bash
export WANDB_PROJECT=YOUR_WANDB_PROJECT_NAME
export WANDB_ENTITY=YOUR_WANDB_ENTITY_NAME
python -m allenact.main training/online/siglip_vitb_gru_rgb_augment_objectnav --callbacks wandb_logging_callback
```

To evaluate, use the following command:

```bash
python -m allenact.main training/online/siglip_vitb_gru_rgb_augment_objectnav -c /path/to/checkpoint --eval
```

You can find the checkpoint in the `experiment_output` directory.

