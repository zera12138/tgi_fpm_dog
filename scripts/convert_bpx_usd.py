"""Reuse the local Isaac asset importer, preserving BPX MJCF physics."""
from pathlib import Path
import sys, xml.etree.ElementTree as ET, os, tempfile, shutil
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tgi/src'))
from tgi.robot import MJCF, USD

def main():
    # Importers need the same isolated USD runtime as Isaac Lab's kitless Newton launcher.
    from isaaclab.sim import SimulationCfg
    from isaaclab_newton.physics import NewtonCfg
    from isaaclab.app import launch_simulation
    cfg=SimulationCfg(device='cpu',physics=NewtonCfg())
    with launch_simulation(cfg,{'headless':True,'device':'cpu'}):
        from pxr import Plug
        import isaacsim.asset.importer.utils.impl as utils
        from isaacsim.asset.importer.mjcf import MJCFImporter,MJCFImporterConfig
        Plug.Registry().RegisterPlugins(os.path.join(os.path.dirname(utils.__file__),'schemas'))
        tree=ET.parse(MJCF);world=tree.getroot().find('worldbody')
        for node in list(world):
            if node.tag!='body':world.remove(node)
        # The asset must contain only the robot; terrain is owned by the environment.
        robot_xml=Path(MJCF).with_name('bpx_robot.xml');tree.write(robot_xml)
        # Import to a fresh directory: the upstream importer appends _1 if output exists.
        # Copy the completed asset tree only after a successful conversion.
        with tempfile.TemporaryDirectory(prefix='bpx-usd-import-') as tmp:
            result=MJCFImporter(MJCFImporterConfig(mjcf_path=str(robot_xml),usd_path=tmp)).import_mjcf()
            result=Path(result)
            if not result.is_file():raise RuntimeError(f'Missing converter output: {result}')
            shutil.copytree(result.parent,Path(USD).parent,dirs_exist_ok=True)
            if not Path(USD).is_file():raise RuntimeError(f'Unexpected converter filename: {result.name}')
        print('BPX USD:',USD)
if __name__=='__main__':main()
