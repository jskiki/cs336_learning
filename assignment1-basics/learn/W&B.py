import wandb

run = wandb.init(project="cs336")
run.log({"loss": 1.23})
run.finish()