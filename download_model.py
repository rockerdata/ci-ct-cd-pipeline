import wandb
import os
import shutil
from pathlib import Path
from config.config import CONFIG



class DownlodModel:
    def __init__(self) -> None:
        pass
    def download_model(self):
        wandb.login(key=CONFIG['WANDB_API_KEY'])
        run = wandb.init()
        artifact = run.use_artifact('rmore/github_actions_wandb_aws_ec2_sklearn/github_wandb:v0', type='model')
        artifact_dir = artifact.download()
        wandb.finish()
        #creating a new directory called pythondirectory
        Path("./models").mkdir(parents=True, exist_ok=True)
        shutil.copy(artifact_dir+"/model.pkl","./models/model.pkl")



if __name__=="__main__":
    obj = DownlodModel()
    obj.download_model()
