from pathlib import Path
import yaml

_project_root = Path(__file__).resolve().parent.parent.parent

with open(_project_root / 'config.yaml') as _f:
    _cfg = yaml.safe_load(_f)

# Expose only the runtime config keys (exclude training defaults)
project_config = {k: v for k, v in _cfg.items() if k != 'training'}

data_root = Path(_cfg.get('data_root', '/data/taveesh/vca'))
mlflow_tracking_uri = f"sqlite:///{data_root / _cfg['mlflow_db']}"
