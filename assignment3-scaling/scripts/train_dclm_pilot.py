"""Submit the DCLM training pilot on Modal and persist its report."""

import json
from pathlib import Path

import modal

DATA_DIR = Path("/data/dclm-pilot")
RUNS_DIR = Path("/data/dclm-pilot-runs")

app = modal.App("cs336-a3-dclm-training-pilot")
volume = modal.Volume.from_name("cs336-a3-dclm")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential")
    .uv_pip_install("numpy")
    .uv_pip_install("torch==2.11.0", index_url="https://download.pytorch.org/whl/cu130")
)
if modal.is_local():
    source = Path(__file__).resolve().parents[1] / "cs336_scaling"
    image = (
        image.add_local_file(source / "__init__.py", "/root/cs336_scaling/__init__.py")
        .add_local_file(
            source / "pilot_training.py", "/root/cs336_scaling/pilot_training.py"
        )
        .add_local_dir(
            source / "torch_model",
            "/root/cs336_scaling/torch_model",
            ignore=["**/__pycache__/**", "**/*.pyc"],
        )
    )


@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="L40S",
    cpu=4,
    memory=8192,
    timeout=900,
    max_containers=1,
)
def train_pilot() -> str:
    from cs336_scaling.pilot_training import run_dclm_pilot

    report = run_dclm_pilot(DATA_DIR)
    call_id = modal.current_function_call_id()
    report_path = RUNS_DIR / f"{call_id}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report["report_path"] = str(report_path)
    result = json.dumps(report, indent=2) + "\n"
    report_path.write_text(result)
    volume.commit()
    return result


@app.local_entrypoint()
def main():
    print(train_pilot.remote(), end="")
