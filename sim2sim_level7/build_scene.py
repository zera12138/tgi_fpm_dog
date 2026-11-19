"""A connected level-7 test yard, in the layout of moe_lant stairs_and_slope.

Training level 7 samples difficulty in [0.7, 0.8); use its midpoint 0.75.
This is a manually navigable course at matching parameter scales, not an SGS tile.
"""
from pathlib import Path
import json, math, sys
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
PROJECT = Path('/home/odroid/bpx-tgi-fpm')
sys.path[:0] = [str(PROJECT/'tgi/src'), str(PROJECT/'fpm/src')]
from tgi.robot import MJCF
from tgi.env.terrains import stone_rects

D = .75
COLORS = [[1., .16, .12, 1.], [.15, .50, 1., 1.], [1., .75, .10, 1.], [.12, .95, .40, 1.]]
FEET = ['FL', 'FR', 'HL', 'HR']


def build():
    dest = HERE/'scene'; dest.mkdir(exist_ok=True)
    tree = ET.parse(MJCF); root = tree.getroot(); root.set('model', 'BPX_TGI_Level7_Gamepad')
    for mesh in root.findall('./asset/mesh'):
        mesh.set('file', str((Path(MJCF).parent/mesh.get('file')).resolve()))
    asset = root.find('asset'); world = root.find('worldbody')
    for light in list(world.findall('light')): world.remove(light)
    ET.SubElement(world, 'light', pos='2 -3 9', dir='-.1 .2 -1', diffuse='.8 .8 .8', castshadow='true')
    ET.SubElement(world, 'light', pos='-3 5 7', dir='.2 -.2 -1', diffuse='.35 .35 .35', castshadow='false')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='1440', offheight='900')
    ET.SubElement(visual, 'headlight', ambient='.35 .35 .35', diffuse='.45 .45 .45', specular='.1 .1 .1')
    ET.SubElement(visual, 'quality', shadowsize='4096')
    ET.SubElement(visual, 'map', znear='.02', zfar='100')
    ET.SubElement(root, 'statistic', center='4 0 .5', extent='17')
    ET.SubElement(asset, 'texture', name='yard_sky', type='skybox', builtin='gradient', rgb1='.20 .31 .45', rgb2='.75 .83 .88', width='512', height='3072')
    ET.SubElement(asset, 'texture', name='yard_grid', type='2d', builtin='checker', rgb1='.23 .28 .32', rgb2='.29 .34 .38', width='512', height='512')
    ET.SubElement(asset, 'material', name='yard_floor', texture='yard_grid', texrepeat='1 1', texuniform='true', reflectance='.04')
    ET.SubElement(asset, 'material', name='step_even', rgba='.74 .79 .83 1', specular='.15', shininess='.12')
    ET.SubElement(asset, 'material', name='step_odd', rgba='.62 .68 .73 1', specular='.15', shininess='.12')
    ground = world.find("geom[@name='ground']")
    ground.set('pos', '0 0 -.8'); ground.set('rgba', '.09 .12 .15 1'); ground.set('group', '0')

    def box(name, x0, x1, y0, y1, top, bottom, material=None, rgba=None, collision=True):
        attrs = dict(name=name, type='box', pos=f'{(x0+x1)/2} {(y0+y1)/2} {(top+bottom)/2}',
                     size=f'{(x1-x0)/2} {(y1-y0)/2} {(top-bottom)/2}', group='0' if collision else '4')
        if material: attrs['material'] = material
        if rgba: attrs['rgba'] = rgba
        if not collision: attrs.update(contype='0', conaffinity='0')
        return ET.SubElement(world, 'geom', **attrs)

    # Flat connected aprons around two recessed test lanes. No invisible floor under gaps.
    xb = [-8., 2., 9., 16.]; yb = [-10., -6.5, -3., 3., 6.5, 10.]
    for i in range(3):
        for j in range(5):
            if i == 1 and j in (1, 3): continue
            box(f'apron_{i}_{j}', xb[i], xb[i+1], yb[j], yb[j+1], 0., -.15, 'yard_floor')
    rise, tread, n = .03+D*(.16-.03), .275, 8
    top = rise*n; edge = 2.+n*tread; platform_end = edge+1.5
    for i in range(n):
        x = 2.+i*tread
        box(f'stair_{i+1}', x, x+tread, -2., 2., (i+1)*rise, -.05, f'step_{"even" if i%2==0 else "odd"}')
        box(f'stair_edge_{i+1}', x, x+.018, -2., 2., (i+1)*rise+.001, (i+1)*rise, rgba='.95 .65 .14 1', collision=False)
    box('top_platform', edge, platform_end, -2., 2., top, -.05, 'step_even')
    slope = .55*D; theta = math.atan(slope); run = top/slope; thick = .15
    ramp_end = platform_end+run
    ET.SubElement(world, 'geom', name='ramp', type='box', group='0', material='step_odd',
                  size=f'{math.hypot(run,top)/2} 2 {thick/2}', euler=f'0 {theta} 0',
                  pos=f'{(platform_end+ramp_end)/2-math.sin(theta)*thick/2} 0 {top/2-math.cos(theta)*thick/2}')

    # Reuse the actual BPX training stone generator and its default easing/jitter.
    rng = np.random.default_rng(7)
    rects = stone_rects(D**1.5, (7., 3.5), rng, width_range=(.6,.2), gap_range=(.02,.18),
                        height_limit=(0.,.05), width_jitter=.05, gap_jitter=.03)
    for i, (x0,x1,y0,y1,z) in enumerate(rects):
        box(f'stone_{i}', x0+2., x1+2., y0+3., y1+3., z, -.8, rgba='.48 .66 .67 1' if i%2 else '.58 .74 .73 1')
    # Same height amplitude/quantization as training; box cells form a traversable manual-test lane.
    rng = np.random.default_rng(17); amplitude = .01+D*(.06-.01); cell=.25
    box('rough_base', 2.,9.,-6.5,-3.,-.06,-.15,'yard_floor')
    for i,x in enumerate(np.arange(2.,9.,cell)):
        for j,y in enumerate(np.arange(-6.5,-3.,cell)):
            z = np.round(rng.uniform(-amplitude,amplitude)/.01)*.01
            box(f'rough_{i}_{j}', x,x+cell,y,y+cell,float(z),-.06,
                rgba='.52 .46 .38 1' if (i+j)%2 else '.63 .57 .48 1')
    # Lane entrance markings are visual only, so the height map contains collision surfaces only.
    for name, y0,y1,color in [('stairs',-2,2,'.95 .65 .14 1'),('stones',3,6.5,'.22 .76 .72 1'),('rough',-6.5,-3,'.85 .59 .33 1')]:
        box(f'entry_{name}',1.8,1.85,y0,y1,.002,.001,rgba=color,collision=False)
    for name, color in zip(FEET,COLORS):
        b = ET.SubElement(world,'body',name=f'fpm_{name}',mocap='true',pos='0 0 -4')
        ET.SubElement(b,'geom',type='sphere',size='.035',rgba=' '.join(map(str,color)),group='5',contype='0',conaffinity='0')
        ET.SubElement(b,'geom',type='cylinder',size='.008 .07',pos='0 0 .07',rgba=' '.join(map(str,color)),group='5',contype='0',conaffinity='0')
    xml = dest/'level7.xml'; ET.indent(tree, space='  '); tree.write(xml, encoding='unicode')
    model=mujoco.MjModel.from_xml_path(str(xml)); data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    # Build the ideal perception map by raycasting the SAME collision scene, not a separate approximation.
    res=.04; origin=np.array([-10.,-12.]); xs=np.arange(-10.,18.001,res);ys=np.arange(-12.,12.001,res)
    h=np.empty((len(xs),len(ys)),np.float32);mask=np.array([1,0,0,0,0,0],np.uint8);gid=np.zeros(1,np.int32)
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            distance=mujoco.mj_ray(model,data,np.array([x,y,5.]),np.array([0.,0.,-1.]),mask,True,-1,gid)
            h[i,j]=5.-distance if distance>=0 else -.8
    np.savez_compressed(dest/'heightmap.npz', heights=h,origin=origin,res=res)
    spawns = [dict(name='stairs approach',xy=[0.,0.],z=0.,yaw=0.),
              dict(name='stairs descent',xy=[edge+.7,0.],z=top,yaw=math.pi),
              dict(name='slope ascent',xy=[ramp_end+1.,0.],z=0.,yaw=math.pi),
              dict(name='stepping stones',xy=[.4,4.75],z=0.,yaw=0.),
              dict(name='rough ground',xy=[.4,-4.75],z=0.,yaw=0.)]
    manifest=dict(level=7,level_interval=[.7,.8],difficulty=D,stairs_rise=rise,stairs_tread=tread,
                  stair_count=n,platform_height=top,slope_gradient=slope,slope_degrees=math.degrees(theta),
                  rough_amplitude=amplitude,stone_effective_difficulty=D**1.5,stone_seed=7,
                  stone_width_nominal=.6-.4*D**1.5,stone_gap_nominal=.02+.16*D**1.5,
                  stone_width_jitter=.05,stone_gap_jitter=.03,pit_depth=.8,
                  heightmap_res=res,spawns=spawns,foot_colors=dict(zip(FEET,COLORS)),
                  note='Custom connected course with training level-7 parameter scales; not an exact training tile. Rough surface uses .25m box cells. Stone pit depth .8m is fixed within training range.')
    (dest/'scene.json').write_text(json.dumps(manifest,indent=2))
    print(f'Built {xml}: {model.ngeom} geoms, map {h.shape}, stair rise {rise:.4f}m, slope {math.degrees(theta):.2f}deg')

if __name__=='__main__': build()
