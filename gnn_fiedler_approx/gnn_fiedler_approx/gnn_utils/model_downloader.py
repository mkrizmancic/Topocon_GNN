import os
import pathlib
import wandb
import torch

from gnn_fiedler_approx.algebraic_connectivity_script import generate_model

HPC_MODEL_DIR = pathlib.Path("/lustre/home/mkrizman/Topocon_GNN/gnn_fiedler_approx/models/")
LOC_MODEL_DIR = pathlib.Path(__file__).parents[2] / "models"

def download_model(run, name):
    model_dict_path = LOC_MODEL_DIR / f"{run.id}_best_model.pth"
    if not model_dict_path.exists():
        os.system(f"scp mkrizman@login-gpu.hpc.srce.hr:{HPC_MODEL_DIR}/{run.id}_best_model.pth {LOC_MODEL_DIR}/")
    else:
        print(f"Model file {model_dict_path} already exists, skipping download.")

    model = generate_model(
        architecture=run.config["architecture"],
        in_channels=len(run.config["selected_features"]),
        hidden_channels=run.config["hidden_channels"],
        gnn_layers=run.config["gnn_layers"],
        mlp_layers=run.config["mlp_layers"],
        pool=run.config["pool"],
        jk=run.config["jk"],
        dropout=run.config["dropout"],
        norm=run.config["norm"],
        act=run.config["activation"],
    )
    model.load_state_dict(torch.load(model_dict_path)["model_state_dict"])

    model.save(LOC_MODEL_DIR / f"{name}.pth")


if __name__ == "__main__":
    model_mapping = {
        "cu25ck52": "dist-32",
        "rwto26el": "dist-64",
        "9kz9vxhs": "full"
    }

    api = wandb.Api()
    for run_id, model_name in model_mapping.items():
        run = api.run(f"marko-krizmancic/gnn_fiedler_approx_v3/{run_id}")
        download_model(run, model_name)
