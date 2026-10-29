"""Train-only geometric/statistical calibration of one frozen expert.

These are controlled calibration hypotheses, not novel SNN-specific theorems.
"""
import torch
from torch import nn
import torch.nn.functional as F

VARIANTS=('none','depth_scale','depth_affine','normal_rotation','normal_tangent','seg_temperature','seg_bias','combined')

def depth_coefficients(stats,shift=True):
    n,x,y,xx,xy=[float(stats[k]) for k in ('n','x','y','xx','xy')]
    if n<=0:raise ValueError('no valid training depth')
    if not shift:return max(0.,xy/xx) if xx>0 else 0.,0.
    variance=xx-x*x/n
    scale=max(0.,(xy-x*y/n)/variance) if variance>1e-12 else 0.
    return scale,(y-scale*x)/n

def proper_rotation(cross):
    h=torch.as_tensor(cross,dtype=torch.float64)
    u,_,vh=torch.linalg.svd(h)
    fix=torch.eye(3,dtype=h.dtype);fix[-1,-1]=torch.linalg.det(u@vh)
    return (u@fix@vh).float()

def tangent_bases():
    values=torch.tensor([
      [[0,0,0],[0,0,-1],[0,1,0]],[[0,0,1],[0,0,0],[-1,0,0]],[[0,-1,0],[1,0,0],[0,0,0]],
      [[1,0,0],[0,-1,0],[0,0,0]],[[1,0,0],[0,1,0],[0,0,-2]],
      [[0,1,0],[1,0,0],[0,0,0]],[[0,0,1],[0,0,0],[1,0,0]],[[0,0,0],[0,0,1],[0,1,0]]],dtype=torch.float32)
    return values/values.square().sum((1,2),keepdim=True).sqrt()

class TaskCalibration(nn.Module):
    def __init__(self,num_classes,mode='none',statistics=None):
        super().__init__();self.mode=mode;self.depth=nn.Parameter(torch.tensor([1.,0.]),requires_grad=mode in ('depth_scale','depth_affine','combined'))
        self.register_buffer('rotation',torch.eye(3));self.register_buffer('bases',tangent_bases())
        self.bias=nn.Parameter(torch.zeros(num_classes),requires_grad=mode in ('seg_bias','combined'))
        self.log_temperature=nn.Parameter(torch.zeros(()),requires_grad=mode=='seg_temperature')
        self.direction=nn.Parameter(torch.zeros(11),requires_grad=mode in ('normal_tangent','combined'))
        if statistics is not None:
            if mode in ('depth_scale','depth_affine','combined'):
                self.depth.copy_(torch.tensor(depth_coefficients(statistics['depth'],mode!='depth_scale')))
            if mode in ('normal_rotation','normal_tangent','combined'):
                if statistics['normals_n']<=0:raise ValueError('no valid training normals')
                self.rotation.copy_(proper_rotation(statistics['normal_cross']))
    def parameters_for(self,task):
        if task=='semseg':
            if self.bias.requires_grad:return [self.bias]
            if self.log_temperature.requires_grad:return [self.log_temperature]
        if task=='normals' and self.direction.requires_grad:return [self.direction]
        if task=='depth' and self.depth.requires_grad:return [self.depth]
        return []
    def forward(self,raw,task):
        if task=='semseg':
            if self.mode=='seg_temperature':return raw/self.log_temperature.exp()
            return raw+(self.bias-self.bias.mean())[None,:,None,None] if self.mode in ('seg_bias','combined') else raw
        if task=='depth':
            if self.mode not in ('depth_scale','depth_affine','combined'):return raw
            return (self.depth[0]*raw.clamp_min(1e-9)+self.depth[1]).clamp_min(1e-9)
        if task=='normals':
            if self.mode not in ('normal_rotation','normal_tangent','combined'):return raw
            rotated=torch.einsum('ij,bjhw->bihw',self.rotation,raw)
            if self.mode=='normal_rotation':return rotated
            length=rotated.square().sum(1,keepdim=True).clamp_min(1e-12).sqrt();p=rotated/length
            matrix=torch.einsum('k,kij->ij',self.direction[:8],self.bases)
            delta=torch.einsum('ij,bjhw->bihw',matrix,p)+self.direction[8:][None,:,None,None]
            tangent=delta-(p*delta).sum(1,keepdim=True)*p
            return rotated+length*tangent
        return raw

class FrozenCalibratedExpert(nn.Module):
    def __init__(self,expert,heads,tasks,calibrator,depth_gain=1.25):
        super().__init__();self.expert=expert.eval().requires_grad_(False);self.heads=heads.eval().requires_grad_(False)
        self.tasks=list(tasks);self.calibrator=calibrator;self.depth_gain=depth_gain
    def train(self,mode=True):
        super().train(mode);self.expert.eval();self.heads.eval();return self
    @torch.no_grad()
    def raw_task(self,x,t,context,size):
        task=self.tasks[t]
        if task!='edge':
            y=self.expert.forward_many({t:x},task_context=context)[t]
            x=x+(self.depth_gain if task=='depth' else 1.)*(y-x)
        return F.interpolate(self.heads[task](x),size,mode='bilinear',align_corners=False)
    def forward_task(self,x,t,context,size):return self.calibrator(self.raw_task(x,t,context,size),self.tasks[t])

def fold_linear_parts(heads,calibrator):
    """Return copies with semseg bias/temperature and normal rotation folded.

    Depth affine remains explicit: clamping before a shift need not commute.
    """
    import copy
    heads=copy.deepcopy(heads);remaining=copy.deepcopy(calibrator)
    with torch.no_grad():
        if calibrator.mode=='seg_temperature':
            layer=heads['semseg'].linear_pred;scale=calibrator.log_temperature.exp().to(layer.weight)
            layer.weight.div_(scale);layer.bias.div_(scale);remaining.log_temperature.zero_()
        if calibrator.mode in ('seg_bias','combined'):
            layer=heads['semseg'].linear_pred;layer.bias.add_((calibrator.bias-calibrator.bias.mean()).to(layer.bias));remaining.bias.zero_()
        if calibrator.mode in ('normal_rotation','normal_tangent','combined'):
            layer=heads['normals'].linear_pred;r=calibrator.rotation.to(layer.weight)
            layer.weight.copy_(torch.einsum('ij,jchw->ichw',r,layer.weight));layer.bias.copy_(r@layer.bias)
            remaining.rotation.copy_(torch.eye(3,device=remaining.rotation.device))
    return heads,remaining

def load_calibration_checkpoint(path,heads,device='cpu'):
    import pathlib,sys
    sys.path.insert(1,str(pathlib.Path(__file__).resolve().parent.parent/'CVPR_sprint_20261024'))
    from sprint_model import make_expert
    state=torch.load(path,map_location='cpu');z=torch.load(state['source_checkpoint'],map_location='cpu');s=z['spec']
    teacher=make_expert(state['tasks'],kind=s['kind'],encoding=s['encoding'],width=s.get('width',128))
    teacher.load_state_dict(z['model'],strict=True)
    c=TaskCalibration(state['num_classes'],state['variant'],state['statistics']);c.load_state_dict(state['calibration'],strict=True)
    return FrozenCalibratedExpert(teacher,heads,state['tasks'],c,state['depth_gain']).to(device).eval().requires_grad_(False)

def install_task_calibration(model,path):
    from types import MethodType
    if model.refine_steps!=1 or model.stream_policy not in (None,'reset') or model.stream_persist_frames or '3ddet' in model.tasks:
        raise ValueError('only independent-image single-refinement dense prediction')
    model.calibrated_expert=load_calibration_checkpoint(path,model.heads,next(model.parameters()).device)
    from intent_multiplexed import install_prompt_context_capture
    if list(model.tasks)!=model.calibrated_expert.tasks:raise ValueError('task ordering mismatch')
    if not hasattr(model.backbone,'_multiplex_context_hook'):install_prompt_context_capture(model.backbone)
    def execute(self,memory,requested_tasks=None,spike_states=None,return_spike_states=False):
        if spike_states or return_spike_states:raise ValueError('state resets per image')
        tasks=self._normalize_requested_tasks(requested_tasks);bm=memory['backbone_memory'];features,_=self.backbone.decode_visual_memory(bm,requested_tasks=tasks)
        context=bm.get('prompt_context');size=self.target_size if self.target_size is not None else memory['img_size'];result={}
        for task in tasks:
            t=self.task_to_index[task];x=features[task]
            if self.task_controller_gate is not None:x=self.task_controller_gate(x,t)
            result[task]=self.calibrated_expert.forward_task(x,t,None if context is None else context[:,t:t+1],size)
        return result
    def single(self,image,requested_tasks=None,force_persist=False):
        if force_persist:raise ValueError('physical streams are a separate experiment')
        return self.execute_task(self.encode_visual_memory(image),requested_tasks=requested_tasks)
    model.execute_task=MethodType(execute,model);model._forward_single_frame=MethodType(single,model)
    return model
