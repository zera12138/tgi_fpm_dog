"""Build a BPX MJCF from the supplied URDF, preserving inertias and collision geometry.

This converter is deliberately limited to the fixed/revolute joints and geometries present
in BPX; unsupported elements fail explicitly. No system identification is performed.
"""
from pathlib import Path
import sys, os, json, hashlib
import xml.etree.ElementTree as ET
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tgi/src'))
from tgi.robot import *

def nums(s): return np.fromstring(s, sep=' ')
def fmt(x): return ' '.join(f'{float(v):.12g}' for v in x)
def rotation(rpy):
    r,p,y=nums(rpy); cr,sr=np.cos(r),np.sin(r); cp,sp=np.cos(p),np.sin(p); cy,sy=np.cos(y),np.sin(y)
    return np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],[sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])
def quat(rpy):
    r,p,y=nums(rpy)/2; cr,sr=np.cos(r),np.sin(r); cp,sp=np.cos(p),np.sin(p); cy,sy=np.cos(y),np.sin(y)
    return fmt([cr*cp*cy+sr*sp*sy,sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy])
def origin(node):
    o=node.find('origin')
    return {'pos':o.get('xyz','0 0 0') if o is not None else '0 0 0',
            'quat':quat(o.get('rpy','0 0 0') if o is not None else '0 0 0')}

def main():
    src=Path(URDF); urdf=ET.parse(src).getroot(); dst=Path(MJCF)
    m=ET.Element('mujoco',model='bpx');ET.SubElement(m,'compiler',angle='radian',fusestatic='false',inertiafromgeom='false',autolimits='true')
    ET.SubElement(m,'option',timestep=str(PHYSICS_DT),integrator='implicitfast',gravity='0 0 -9.81')
    defaults=ET.SubElement(m,'default');ET.SubElement(defaults,'joint',armature=str(ARMATURE),damping=str(JOINT_DAMPING),frictionloss=str(JOINT_FRICTION))
    ET.SubElement(defaults,'geom',friction='0.8 0.005 0.0001',condim='3',solref='0.01 1')
    assets=ET.SubElement(m,'asset'); meshes={}
    def geometry(parent,g,atts):
        shape=list(g)[0]; tag=shape.tag
        if tag=='mesh':
            meshpath=(src.parent/shape.get('filename')).resolve()
            key=(str(meshpath),shape.get('scale','1 1 1'))
            if key not in meshes:
                name=f'mesh_{len(meshes)}';meshes[key]=name
                ET.SubElement(assets,'mesh',name=name,file=os.path.relpath(meshpath,dst.parent),scale=key[1])
            atts.update(type='mesh',mesh=meshes[key])
        elif tag=='box':atts.update(type='box',size=fmt(nums(shape.get('size'))/2))
        elif tag=='cylinder':atts.update(type='cylinder',size=f"{shape.get('radius')} {float(shape.get('length'))/2}")
        elif tag=='sphere':atts.update(type='sphere',size=shape.get('radius'))
        else:raise ValueError(f'Unsupported shape {tag}')
        ET.SubElement(parent,'geom',**atts)
    links={l.get('name'):l for l in urdf.findall('link')};children={};child_names=set()
    for j in urdf.findall('joint'):
        children.setdefault(j.find('parent').get('link'),[]).append(j);child_names.add(j.find('child').get('link'))
    roots=set(links)-child_names
    if roots!={'torso'}:raise ValueError(roots)
    world=ET.SubElement(m,'worldbody');ET.SubElement(world,'light',pos='0 0 3',dir='0 0 -1')
    ET.SubElement(world,'geom',name='ground',type='plane',size='0 0 0.1',rgba='0.25 0.3 0.35 1')
    def body(parent,name,j=None):
        atts={'name':name,**(origin(j) if j is not None else {'pos':f'0 0 {NOMINAL_HEIGHT}'})}
        b=ET.SubElement(parent,'body',**atts);l=links[name]
        if j is None:ET.SubElement(b,'freejoint',name='root')
        elif j.get('type')=='revolute':
            lim=j.find('limit');ET.SubElement(b,'joint',name=j.get('name'),type='hinge',axis=j.find('axis').get('xyz'),range=f"{lim.get('lower')} {lim.get('upper')}")
        elif j.get('type')!='fixed':raise ValueError(j.get('type'))
        inert=l.find('inertial')
        if inert is not None:
            a={k:float(v) for k,v in inert.find('inertia').attrib.items()}
            I=np.array([[a['ixx'],a['ixy'],a['ixz']],[a['ixy'],a['iyy'],a['iyz']],[a['ixz'],a['iyz'],a['izz']]])
            io=inert.find('origin');R=rotation(io.get('rpy','0 0 0') if io is not None else '0 0 0');I=R@I@R.T
            ET.SubElement(b,'inertial',pos=origin(inert)['pos'],mass=inert.find('mass').get('value'),fullinertia=fmt([I[0,0],I[1,1],I[2,2],I[0,1],I[0,2],I[1,2]]))
        for kind in ('collision','visual'):
            for idx,g in enumerate(l.findall(kind)):
                ga={'name':f'{name}_{kind}_{idx}',**origin(g),'group':'3' if kind=='collision' else '2'}
                if kind=='visual':ga.update(contype='0',conaffinity='0',rgba='0.25 0.3 0.4 1')
                geometry(b,g.find('geometry'),ga)
        if name in FOOT_BODIES:
            leg=LEGS[FOOT_BODIES.index(name)];ET.SubElement(b,'site',name=f'{leg}_foot',pos=fmt(FOOT_OFFSET),size='0.006',rgba='1 0.2 0.1 1')
        for c in children.get(name,[]):body(b,c.find('child').get('link'),c)
    body(world,'torso')
    act=ET.SubElement(m,'actuator')
    for n in JOINT_NAMES:ET.SubElement(act,'motor',name=n,joint=n,ctrlrange=f'-{EFFORT_LIMIT} {EFFORT_LIMIT}')
    key=ET.SubElement(m,'keyframe');ET.SubElement(key,'key',name='stand',qpos=fmt([0,0,NOMINAL_HEIGHT,1,0,0,0,*DEFAULT_Q]))
    ET.indent(m);ET.ElementTree(m).write(dst,encoding='unicode')
    import mujoco
    model=mujoco.MjModel.from_xml_path(str(dst));assert model.nu==NUM_J and model.nq==19
    report={'source_urdf':str(src),'sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'mjcf':str(dst),'mass':float(model.body_mass.sum()),'nq':model.nq,'nv':model.nv,'nu':model.nu,'joints':list(JOINT_NAMES),'feet':list(FOOT_SITES),'note':'URDF mesh collisions use MuJoCo convex hulls; MJCF visuals preserve source meshes. PD/armature are simulation assumptions.'}
    (ROOT/'docs/asset_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
