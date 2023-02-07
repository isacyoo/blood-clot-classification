import optuna

from config import Config
from tuning import tune_nuisance_hyperparameters, produce_study_summary


if __name__ == "__main__":
    
    # Three scientific hyperparameters to be studied
    first_hidden_sizes = [16, 32, 64]
    num_stages = [3, 4]
    num_layers = [2, 3]
    n_trials = 20

    config = Config()
    for hidden_d in first_hidden_sizes:
        for stages in num_stages:
            for layers in num_layers:
                config.model.first_hidden_size = hidden_d
                config.model.num_stages = stages
                config.model.num_layers = layers
                
                study = optuna.create_study(study_name=f"first_hidden_size:{hidden_d}-num_stages:{stages}-num_layers:{layers}", direction="maximize")
                obj_func = lambda trial: tune_nuisance_hyperparameters(trial, config)
                study.optimize(obj_func, n_trials=n_trials, n_jobs=1, gc_after_trial=True)
                
                produce_study_summary(study)