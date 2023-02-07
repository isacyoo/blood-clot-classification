import os

import optuna
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from model import BloodClotClassifier
from data_processing import BloodClotDataModule
from config import Config

# Optuna objective function to tune nuisance hyperparameters

def tune_nuisance_hyperparameters(trial: optuna.trial.Trial, config: Config) -> float:
    
    # model
    config.model.dropout = trial.suggest_float("dropout", 0, 0.5)
    config.model.act = trial.suggest_categorical("act", ["relu", "gelu"])
    
    # optimiser
    config.train.lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    config.train.weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-5, log=True)
    config.train.momentum = trial.suggest_float("momentum", 0, 1)
    
    # image augmentation
    config.data.aug_p = trial.suggest_float("aug_p", 0.5, 1)
    config.data.shift_limit = trial.suggest_float("shift_limit", 0, 0.5)
    config.data.scale_limit = trial.suggest_float("scale_limit", 0, 0.5)
    config.data.rotate_limit = trial.suggest_int("rotate_limit", 0, 30)
    
    
    # Get validation error
    model = BloodClotClassifier(config)
    BloodClotDM = BloodClotDataModule(config)
    es = pl.callbacks.EarlyStopping(monitor="val_auroc", verbose=True, patience = 2, mode="max")
    
    trainer = pl.Trainer(callbacks=[es], accelerator="gpu" if torch.cuda.is_available() else "cpu",
                     enable_model_summary=True, precision=16, gradient_clip_val=1.0,
                    val_check_interval=1.0, max_epochs=-1)
    trainer.fit(model, datamodule=BloodClotDM)
    return es.best_score.item()

# Produces summary for a study, including a csv file for each run, a txt file containing variable importances and a png file of isolated plots of four dominant variables
def produce_study_summary(study, config):
    print(f"Producing summary for {study.study_name}")
    df = study.trials_dataframe()
    if not os.path.exists(f"{config.paths.output_dir}/{study.study_name}"):
        os.mkdir(f"{config.paths.output_dir}/{study.study_name}")
    df.to_csv(f"{config.paths.output_dir}/{study.study_name}/trials.csv")
    var_importances = list(optuna.importance.get_param_importances(study).items())
    with open(f"{config.paths.output_dir}/{study.study_name}/variable_importances.txt","w") as f:
        for var, importance in var_importances:
            f.write(f"{var}: {importance}\n")
    
    fig, axis = plt.subplots(2,2)
    
    axis[0, 0].scatter(df[f"params_{var_importances[0][0]}"], df["value"])
    axis[0, 0].set_xlabel(var_importances[0][0])
    axis[0, 0].set_ylabel("val_AUROC")

    axis[0, 1].scatter(df[f"params_{var_importances[1][0]}"], df["value"])
    axis[0, 1].set_xlabel(var_importances[1][0])
    axis[0, 1].set_ylabel("val_AUROC")
    
    axis[1, 0].scatter(df[f"params_{var_importances[2][0]}"], df["value"])
    axis[1, 0].set_xlabel(var_importances[2][0])
    axis[1, 0].set_ylabel("val_AUROC")
    
    axis[1, 1].scatter(df[f"params_{var_importances[3][0]}"], df["value"])
    axis[1, 1].set_xlabel(var_importances[3][0])
    axis[1, 1].set_ylabel("val_AUROC")
    
    fig.tight_layout()
    plt.savefig(f"{config.paths.output_dir}/{study.study_name}/var_importance_plot.png")
    plt.show()
    