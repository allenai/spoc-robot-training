# [SPOC 🖖: Imitating Shortest Paths in Simulation Enables Effective Navigation and Manipulation in the Real World](https://spoc-robot.github.io/)

[![License](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


This repository contains the code and data for the paper "Imitating Shortest Paths in Simulation Enables Effective Navigation and Manipulation in the Real World".

## 🐍 Setting up the Python environment 🐍

### 🛠 Local installation 🛠

```bash
pip install -r requirements.txt
pip install --extra-index-url https://ai2thor-pypi.allenai.org ai2thor==0+5d0ab8ab8760eb584c5ae659c2b2b951cab23246
```

### 🐳 Docker 🐳 

Please see the [README.md](docker/README.md) in the `docker` directory for instructions on how to build and run the docker image.

## 🏋️ Training 🏋️

### 📥 Downloading the training data 📥

We have two different types of training data, `fifteen` and `all`. The `fifteen` type has the agent navigating and fetching one of fifteen possible object types, the `all` type has the agent navigating to and fetching any object type (of many hundreds). To download the training data for the `fifteen` type, run the following command:  

```bash
python -m scripts.download_training_data --save_dir /your/local/save/dir --types fifteen
```

and change `fifteen` to `all` to download the `all` type training data. By default, you will download data corresponding to all 10 task types (ObjectNav, Pickup, Fetch, etc); if you'd only like to download data for a subset of task types, look at the the `--task_types` flag. 

#### 📁 Dataset format 📁

Once you run the above command, you will have a directory structure that looks like this
```
/your/local/save/dir/<fifteen OR all>_type
    <TASK_TYPE>
        house_id_to_sub_house_id_train.json # This file contains a mapping that's needed for train data loading
        house_id_to_sub_house_id_val.json   # This file contains a mapping that's needed for val data loading
        train
            <HOUSEID>
                hdf5_sensors.hdf5 -- containing all the sensors that are not videos
                    <EPISODE_NUMBER>
                        <SENSOR_NAME>
                raw_navigation_camera__<EPISODE_NUMBER>.mp4
                raw_manipulation_camera__<EPISODE_NUMBER>.mp4
        val
            # As with train
```


The `raw_*_camera_*.mp4` files contain videos of the agent's trajectories in the environments. The `hdf5_sensors.hdf5` contains all sensors values that are not camera frames. Below is the definition of all the sensors found in the `hdf5_sensors.hdf5` files:


| Sensor Name                        | Definition                                                                                                                          |
|------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| `an_object_is_in_hand`             | Indicates whether an object is currently being held in hand.                                                                        |
| `house_index`                      | Indicates the index of the house in the dataset.                                                                                    |
| `hypothetical_task_success`        | Indicates whether the task will be successful if the agent would have issued done at this step.                                     |
| `last_action_is_random`            | Indicates if the most recent action taken was chosen randomly.                                                                      |
| `last_action_str`                  | A string representation of the last action performed.                                                                               |
| `last_action_success`              | Indicates whether the last action performed was successful.                                                                         |
| `last_agent_location`              | Indicates agent's location in the world coordinate frame.                                                                           |
| `manip_accurate_object_bbox`       | Indicates the bounding box for the target object in the manipulation camera, based on one method of calculation in simulation.      |
| `manip_task_relevant_object_bbox`  | Indicates the bounding box for the target object in the manipulation camera, based on a second method of calculation in simulation. |
| `minimum_l2_target_distance`       | The minimum Euclidean (L2) distance to the target object or location.                                                               |
| `minimum_visible_target_alignment` | Measures the minimum degree the agent needs to turn to center the object in the navigation camera frame (if object is visible).     |
| `nav_accurate_object_bbox`         | Indicates the bounding box for the target object in the navigation camera, based on one method of calculation in simulation.        |
| `nav_task_relevant_object_bbox`    | Indicates the bounding box for the target object in the navigation camera, based on a second method of calculation in simulation.   |
| `relative_arm_location_metadata`   | Arm proprioceptive, relative location of the arm in the agent's coordinate frame.                                                   |
| `room_current_seen`                | Indicates whether this room has been seen before or not.                                                                            |
| `rooms_seen`                       | Count of rooms that have been visited by the agent.                                                                                 |
| `templated_task_spec`              | A dictionary of the task information, which can be used to generate the natural language description of the task.                   |
| `visible_target_4m_count`          | The count of targets visible within a 4-meter radius or distance.                                                                   |

#### 🤖 Actions 🤖
Our discrete action space is as follows:

- **move_ahead** (`m`): Move the agent’s base forward by 0.2 meters.
- **move_back** (`b`): Move the agent’s base backward by 0.2 meters.
- **rotate_left** (`l`): Rotate the agent’s base left by 30°.
- **rotate_right** (`r`): Rotate the agent’s base right by 30°.
- **rotate_left_small** (`ls`): Rotate the agent’s base left by 6°.
- **rotate_right_small** (`rs`): Rotate the agent’s base right by 6°.
- **move_arm_up** (`yp`): Move the arm up by 0.1 meters.
- **move_arm_down** (`ym`): Move the arm down by 0.1 meters.
- **move_arm_out** (`zp`): Extend the arm outward by 0.1 meters.
- **move_arm_in** (`zm`): Retract the arm inward by 0.1 meters.
- **move_arm_up_small** (`yps`): Move the arm up by 0.02 meters.
- **move_arm_down_small** (`yms`): Move the arm down by 0.02 meters.
- **move_arm_out_small** (`zps`): Extend the arm outward by 0.02 meters.
- **move_arm_in_small** (`zms`): Retract the arm inward by 0.02 meters.
- **wrist_open** (`wp`): Rotate the wrist counterclockwise by 10°.
- **wrist_close** (`wm`): Rotate the wrist clockwise by 10°.
- **end** (`end`): Signal the end of a task.
- **sub_done** (`sub_done`): Mark a sub-task as complete.
- **pickup** (`p`): Initiate a grasp action to pick up an object.
- **dropoff** (`d`): Execute a release action to drop an object.


#### 💪 Running training 💪

For training commands refer to the [TRAINING_README.md](TRAINING_README.md) file.

## 📊 Evaluation 📊

In order to run evaluation you'll need:

1. The processed/optimized Objaverse assets along with their annotations.
2. The set of ProcTHOR-Objaverse houses you'd like to evaluate on.
3. A trained model checkpoint.

Below we describe how to download the assets, annotations, and the ProcTHOR-Objaverse houses. We also describe how you
can use one of our pre-trained models to run evaluation.

### 💾 Downloading assets, annotations, and houses 💾

#### 📦 Downloading optimized Objaverse assets and annotations 📦

Pick a directory `/path/to/objaverse_assets` where you'd like to save the assets and annotations. Then run the following commands:

```bash
python -m objathor.dataset.download_annotations --version 2023_07_28 --path /path/to/objaverse_assets
python -m objathor.dataset.download_assets --version 2023_07_28 --path /path/to/objaverse_assets
```

These will create the directory structure:
```
/path/to/objaverse_assets
    2023_07_28
        annotations.json.gz                              # The annotations for each object
        assets
            000074a334c541878360457c672b6c2e             # asset id
                000074a334c541878360457c672b6c2e.pkl.gz
                albedo.jpg
                emission.jpg
                normal.jpg
                thor_metadata.json
            ... #  39663 more asset directories
```

#### 🏠 Downloading ProcTHOR-Objaverse houses 🏠

Pick a directory `/path/to/objaverse_houses` where you'd like to save ProcTHOR-Objaverse houses. Then run: 
```bash
python -m scripts.download_objaverse_houses --save_dir /path/to/objaverse_houses --subset val
```
to download the validation set of houses as `/path/to/objaverse_houses/val.jsonl.gz`.
You can also change `val` to `train` to download the training set of houses.

#### 🛣 Setting environment variables 🛣

Next you need to set the following environment variables:
```bash
export OBJAVERSE_HOUSES_DIR=/path/to/objaverse_houses
export OBJAVERSE_DATA_DIR=/path/to/objaverse_assets
```

You can see some usage examples for the downloaded data in
[this jupyter notebook](how_to_use_data.ipynb).

### 🚀 Running evaluation with a pretrained model 🚀

For evaluation commands refer to the [TRAINING_README.md](TRAINING_README.md) file.

## 📚 Attribution 📚

We use Open English WordNet 2022 dataset in our work, attribution to Princeton WordNet and the Open English WordNet team. 
See the [english-wordnet](https://github.com/globalwordnet/english-wordnet) repository for more details.

## 📝 Cite us 📝

```bibtex
@article{spoc2023,        
    author    = {Kiana Ehsani, Tanmay Gupta, Rose Hendrix, Jordi Salvador, Luca Weihs, Kuo-Hao Zeng, Kunal Pratap Singh, Yejin Kim, Winson Han, Alvaro Herrasti, Ranjay Krishna, Dustin Schwenk, Eli VanderBilt, Aniruddha Kembhavi},
    title     = {Imitating Shortest Paths in Simulation Enables Effective Navigation and Manipulation in the Real World},
    journal   = {arXiv},
    year      = {2023},
    eprint    = {2312.02976},
}
```
