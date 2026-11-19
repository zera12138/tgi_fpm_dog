"""Read-only report; does not install packages or start training."""
import sys,json,importlib.metadata as md,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tgi/src'),str(ROOT/'fpm/src')]
import torch
from tgi.robot import MJCF,USD,INTERFACE
packages={}
for name in ('torch','isaaclab','isaaclab-newton','newton','mujoco','onnx','onnxruntime'):
    try:packages[name]=md.version(name)
    except md.PackageNotFoundError:packages[name]=None
report={'python':sys.executable,'packages':packages,'interface':INTERFACE,'cuda_available':torch.cuda.is_available(),'cuda_device_count':torch.cuda.device_count(),'mjcf_exists':Path(MJCF).is_file(),'usd_exists':Path(USD).is_file(),'isaaclab_source':importlib.util.find_spec('isaaclab').origin}
report['training_prerequisites_detected']=report['cuda_available'] and report['mjcf_exists'] and report['usd_exists'] and all(packages[k] for k in ('isaaclab','isaaclab-newton','newton'))
print(json.dumps(report,indent=2))
