"""Controlled last-layer adaptation of frozen event/continuous experts."""
import copy,pathlib,sys,torch
from torch import nn
import torch.nn.functional as F
sys.path.insert(1,str(pathlib.Path(__file__).resolve().parent/'CVPR_calibration_20261027'))
from task_calibration import TaskCalibration
sys.path.insert(1,'/root/autodl-tmp/TaskPrompter_artifacts/CVPR_prediction_20261027')
from prediction_fusion import unit

VARIANTS={f'{family}_{mode}':dict(family=family,head=mode in ('head','head_calibration'),calibration=mode in ('calibration','head_calibration')) for family in ('event','charge') for mode in ('none','calibration','head','head_calibration')}

class HeadTune(nn.Module):
    def __init__(self,expert,heads,tasks,spec):
        super().__init__();self.expert=expert.eval().requires_grad_(False)
        self.heads=copy.deepcopy(heads).eval().requires_grad_(False);self.tasks=list(tasks);self.spec=spec;self.stats={}
        if spec['head']:
            for task in tasks:self.heads[task].linear_pred.requires_grad_(True)
        self.calibrator=TaskCalibration(heads['semseg'].linear_pred.out_channels,'combined' if spec['calibration'] else 'none')
        self.anchors=nn.ModuleDict()
        for task in tasks:
            reference=nn.Module()
            for name,value in self.heads[task].linear_pred.named_parameters():reference.register_buffer(name,value.detach().clone())
            self.anchors[task]=reference
    def train(self,mode=True):
        super().train(mode);self.expert.eval();self.heads.eval();return self
    def trainable_for(self,task):return [p for p in self.heads[task].parameters() if p.requires_grad]+self.calibrator.parameters_for(task)
    def regularizer(self,task):
        layer=self.heads[task].linear_pred;result=next(layer.parameters()).sum()*0
        if self.spec['head']:
            for k,v in layer.named_parameters():
                ref=getattr(self.anchors[task],k);result=result+(v-ref).square().mean()/ref.square().mean().clamp_min(1e-4)
        return result
    def forward_task(self,x,t,context,size,**kwargs):
        task=self.tasks[t]
        with torch.no_grad():
            if task!='edge':
                y=self.expert.forward_many({t:x},task_context=context)[t]
                x=x+(1.25 if task=='depth' else 1.)*(y-x)
        raw=F.interpolate(self.heads[task](x.detach()),size,mode='bilinear',align_corners=False)
        out=self.calibrator(raw,task)
        if task=='semseg':out=F.log_softmax(out,1)
        elif task=='normals':out=unit(out)
        elif task=='depth':out=out.clamp_min(1e-9)
        return out
    def learned_state(self):return dict(heads=self.heads.state_dict(),calibration=self.calibrator.state_dict(),anchors=self.anchors.state_dict())
    def restore(self,state):
        self.heads.load_state_dict(state['heads'],strict=True);self.calibrator.load_state_dict(state['calibration'],strict=True)
        # Older runs constructed the same anchors from the unchanged base heads.
        if 'anchors' in state:self.anchors.load_state_dict(state['anchors'],strict=True)

def load_head_checkpoint(path,heads,device='cpu'):
    sys.path.insert(1,str(pathlib.Path(__file__).resolve().parent.parent/'CVPR_sprint_20261024'))
    from sprint_model import make_expert
    z=torch.load(path,map_location='cpu');a=torch.load(z['source_checkpoint'],map_location='cpu');s=a['spec']
    expert=make_expert(z['tasks'],kind=s['kind'],encoding=s['encoding'],width=s.get('width',128));expert.load_state_dict(a['model'],strict=True)
    network=HeadTune(expert,heads,z['tasks'],z['spec']);network.restore(z['learned']);return network.to(device).eval().requires_grad_(False)

def install_head_execution(model,path,loader=None):
    from types import MethodType
    if model.refine_steps!=1 or model.stream_policy not in (None,'reset') or model.stream_persist_frames:raise ValueError('independent images only')
    network=(load_head_checkpoint if loader is None else loader)(path,model.heads,next(model.parameters()).device)
    if list(model.tasks)!=network.tasks:raise ValueError('task ordering mismatch')
    model.head_tuning=network
    from intent_multiplexed import install_prompt_context_capture
    if not hasattr(model.backbone,'_multiplex_context_hook'):install_prompt_context_capture(model.backbone)
    def execute(self,memory,requested_tasks=None,spike_states=None,return_spike_states=False):
        if spike_states or return_spike_states:raise ValueError('independent images only')
        requested=self._normalize_requested_tasks(requested_tasks);bm=memory['backbone_memory'];features,_=self.backbone.decode_visual_memory(bm,requested_tasks=requested)
        size=self.target_size if self.target_size is not None else memory['img_size'];context=bm.get('prompt_context');out={}
        for task in requested:
            t=self.task_to_index[task];x=features[task]
            if self.task_controller_gate is not None:x=self.task_controller_gate(x,t)
            out[task]=self.head_tuning.forward_task(x,t,None if context is None else context[:,t:t+1],size)
        return out
    def single(self,image,requested_tasks=None,force_persist=False):
        if force_persist:raise ValueError('independent images only')
        return self.execute_task(self.encode_visual_memory(image),requested_tasks=requested_tasks)
    model.execute_task=MethodType(execute,model);model._forward_single_frame=MethodType(single,model);return model
