import datetime
import json
import os
import random
import string
from typing import Dict

import lightning.pytorch as pl
import wandb
import yaml
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities.rank_zero import rank_zero_only


class LoggerInfo:
    pass


class LocalWandbLogger(pl.loggers.wandb.WandbLogger):
    def __init__(self, project, entity, name, save_dir, config, log_model):
        # super().__init__()
        self.logger_info = LoggerInfo()
        self.logger_info.project = project
        self.logger_info.entity = entity
        self.logger_info.name = name
        self.logger_info.save_dir = save_dir

        self.logger_info.config = config
        self.logger_info.log_model = log_model
        self._save_dir = save_dir
        self._log_model = log_model
        random.seed(datetime.datetime.now().microsecond)
        characters = string.ascii_letters + string.digits
        run_id = "".join(random.choice(characters) for _ in range(8))
        self._run_id = run_id
        self._experiment = None

        self._log_model = log_model
        self._logged_model_time: Dict[str, float] = {}
        self._checkpoint_callback = None

    @property
    @rank_zero_only
    def experiment(self):
        if self._experiment is None:
            self._experiment = LocalWandb(
                project=self.logger_info.project,
                entity=self.logger_info.entity,
                name=self.logger_info.name,
                save_dir=self.logger_info.save_dir,
                config=self.logger_info.config,
                log_model=self.logger_info.log_model,
                run_id=self._run_id,
            )
            self._experiment.log_config(self.logger_info.config)

        return self._experiment

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        self.experiment.log(metrics, step)

    @property
    def version(self):
        return self._experiment.id


class LoadLocalWandb:
    def __init__(self, run_id, save_dir) -> None:
        self.run_id = run_id
        self.save_dir = save_dir
        self.config = self.load_config()

    def get_checkpoint(self, ckpt_step):
        path = os.path.join(self.save_dir, self.run_id, f"checkpoint_train_steps={ckpt_step}.ckpt")
        assert os.path.exists(path), "Checkpoint does not exist"
        return path

    def load_config(self):
        json_path = os.path.join(self.save_dir, self.run_id, "config.json")
        yaml_path = os.path.join(self.save_dir, self.run_id, "config.yaml")
        if os.path.isfile(json_path):
            with open(json_path, "r") as f:
                config = json.load(f)
        elif os.path.isfile(yaml_path):
            with open(yaml_path, "r") as f:
                config = yaml.load(f, yaml.BaseLoader)

            def extract_values(data):
                values = {}
                for key, value in data.items():
                    if isinstance(value, dict):
                        if set(value.keys()) == {"desc", "value"}:
                            values[key] = value['value']
                        else:
                            values[key] = extract_values(value)
                return values

            config = extract_values(config)
        else:
            raise ValueError(f"No config (yaml or json) found under {os.path.join(self.save_dir, self.run_id)}")
        return config


class LocalWandb:
    def __init__(
        self,
        project,
        entity,
        name,
        save_dir,
        config=None,
        log_model=None,
        run_id=None,
        exp_name=None,
    ) -> None:
        self.project = project
        self.entity = entity
        self.name = name
        self.save_dir = save_dir
        self.config = config if config is not None else {}
        self.log_model = log_model
        if run_id is None:
            random.seed(datetime.datetime.now().microsecond)
            characters = string.ascii_letters + string.digits
            run_id = "".join(random.choice(characters) for _ in range(8))

        self.run_id = run_id

        self.config["exp_name"] = exp_name if exp_name is not None else self.run_id

        self.full_dir = os.path.join(self.save_dir, self.run_id)
        os.makedirs(self.full_dir, exist_ok=True)
        print("Logging everythin in ", self.full_dir)

    def log_config(self, config):
        with open(os.path.join(self.full_dir, "config.json"), "w") as f:
            json.dump(config, f)

    def write_logs(self, dict_to_log, step=None):
        things_to_log = ""
        if step is not None:
            things_to_log = f"step: {str(step)}\n"

        if type(dict_to_log) == dict:
            for key, value in dict_to_log.items():
                if type(value) == wandb.Table:
                    new_value = LocalTable.get_local_table_from_wandb_table(value)
                    value = new_value
                things_to_log += f"{str(key)}: {str(value)}\n"
        elif type(dict_to_log) == str:
            things_to_log += dict_to_log
        else:
            things_to_log += str(dict_to_log)

        with open(os.path.join(self.full_dir, "logs.txt"), "a") as f:
            f.write(str(things_to_log))
            print(str(things_to_log))
            f.write("\n")

    @property
    def id(self):
        return self.run_id

    def download_artifact(self, artifact_name, save_dir):
        assert os.path.exists(save_dir)

        pass

    def log_artifact(self, artifact: wandb.Artifact, aliases):
        # name = artifact.name
        # type = artifact.type
        # file = artifact.file
        for file, info in artifact._added_local_paths.items():
            command = f"cp {file} {self.full_dir}"
            self.write_logs(f"executing {command}")
            os.system(command)
        pass

    @staticmethod
    def Table(columns):
        table = LocalTable(columns=columns)
        return table

    @staticmethod
    def Video(video_path):
        # print('Video is saved in the path:', video_path)
        return video_path

    @staticmethod
    def Image(image_path):
        # print('Image is saved in the path:', image_path)
        return image_path

    def log(self, dict_to_log, step=None):
        self.write_logs(dict_to_log, step)

        pass


class LocalTable:
    columns = []
    rows = []

    @staticmethod
    def get_local_table_from_wandb_table(wandb_table):
        local_table = LocalTable(wandb_table.columns)
        for row in wandb_table.data:
            local_table.add_data(*row)
        return local_table

    def __init__(self, columns):
        self.columns = columns

    def add_data(self, *data):
        self.rows.append(data)

    def __str__(self) -> str:
        tab_str = ", ".join(self.columns) + "\n"
        for row in self.rows:
            tab_str += ", ".join([str(r) for r in row]) + "\n"
        return tab_str

    def get_dataframe(self):
        return str(self)


if __name__ == "__main__":
    tab = LocalTable(["a", "b"])
    tab.add_data(1, 2)
    tab.add_data(3, 4)
    print(str(tab))
