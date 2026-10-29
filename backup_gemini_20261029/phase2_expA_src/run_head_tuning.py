"""Paired, resumable CVPR attribution on existing room-disjoint caches."""
import os,sys,pathlib,argparse,json,time,random,datetime,gc
os.environ['TASKPROMPTER_DB_ROOT']='/root/autodl-tmp/datasets/taskprompter'
sys.path.insert(1,'/root/autodl-tmp/projects_dir/Multi-Task-Transformer/TaskPrompter')
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(1,'/root/autodl-tmp/TaskPrompter_artifacts/CVPR_sprint_20261024')
from sprint_model import legacy,make_expert,TaskRoutedExperts
from head_tuning import HeadTune,VARIANTS
sys.path.insert(1,'/root/autodl-tmp/TaskPrompter_artifacts/CVPR_prediction_20261027')
from prediction_fusion import metric_loss
EXPERTS=VARIANTS
from evaluation.evaluate_utils import PerformanceMeter
from utils.common_config import get_model,get_criterion
from utils.config import create_config
from utils.utils import get_output

ROOT=pathlib.Path(__file__).resolve().parent
ART=pathlib.Path('/root/autodl-tmp/TaskPrompter_artifacts')
parser=argparse.ArgumentParser()
parser.add_argument('--bases',type=int,nargs='+',default=[2038,2039])
parser.add_argument('--seeds',type=int,nargs='+',default=[0,1])
parser.add_argument('--experts',nargs='+',choices=list(EXPERTS),default=list(EXPERTS))
parser.add_argument('--steps',type=int,default=120)
parser.add_argument('--smoke',action='store_true')
parser.add_argument('--mature',action='store_true')
args=parser.parse_args()
assert not args.mature and ((args.smoke and 1<=args.steps<=20) or args.steps==120)
torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.15,0)
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
deadline=datetime.datetime(2026,11,15,16,tzinfo=datetime.timezone.utc).timestamp()
mode='smoke' if args.smoke else 'head120'
OUT=ROOT/mode;OUT.mkdir(exist_ok=True)
scene=json.loads((ART/'TaskPrompter_functional_gradient_20261023/scene_audit.json').read_text())['mapping']

def write(path,data):
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(data));tmp.replace(path)
def emit(phase,**data):
    d=dict(phase=phase,utc=datetime.datetime.utcnow().isoformat(),**data)
    write(OUT/'status.json',d);print(json.dumps(d),flush=True)
def ctor(tasks,spec):
    return make_expert(tasks,kind=spec['kind'],encoding=spec['encoding'],width=spec.get('width',128)).cuda()
def snapshot_rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch']);torch.cuda.set_rng_state_all(r['cuda'])

for base in args.bases:
    d=OUT/('base'+str(base));d.mkdir(exist_ok=True)
    base_root=ART/'CVPR_sprint_20261024/mature_bases'/('base'+str(base))
    receipt=json.loads((base_root/'final_receipt.json').read_text())
    p=create_config(str(base_root/'base.yml'),{'run_mode':'train'});p.backbone_pretrained=False
    model=get_model(p);payload=torch.load(receipt['checkpoint'],map_location='cpu');model.load_state_dict(payload['model'],strict=True);del payload
    model.eval().requires_grad_(False)
    tasks=list(model.tasks);mapping=dict(model.task_to_index);heads=model.heads.cuda().eval();gate=model.task_controller_gate
    if gate is not None:gate.cuda().eval()
    del model;gc.collect()
    cache_root=ART/'CVPR_sprint_20261024/mature_cache'/('base'+str(base))
    train=legacy.DiskEntries(cache_root/'train',634);dev=legacy.DiskEntries(cache_root/'dev',161)
    train_indices=np.random.RandomState(20261025).permutation(634)[:96].tolist()
    room_first={}
    for i in range(161):
        room=scene[dev[i]['image']]
        if room not in room_first:room_first[room]=i
    selected_rooms=np.random.RandomState(20261025).permutation(sorted(room_first))[:32].tolist()
    dev_indices=[room_first[r] for r in selected_rooms];seen=set(selected_rooms)
    assert len(dev_indices)==32 and len(seen)==32
    write(d/'screen_samples.json',dict(train_indices=train_indices,dev_indices=dev_indices,dev_rooms=sorted(seen),selection='fixed random train96; first image from each of 32 fixed-random dev rooms, disjoint from training rooms; selected without metrics'))
    assert not {scene[x] for x in receipt['train_ids']}&{scene[x] for x in receipt['dev_ids']}
    criterion=get_criterion(p).cuda();p.optimizer='adam';p.scheduler='poly';p.max_iter=args.steps;p.optimizer_kwargs=dict(lr=3e-4,weight_decay=1e-6)
    def feature(entry,task):
        x=F.interpolate(entry['features'][task].cuda(),scale_factor=4,mode='bilinear',align_corners=False)
        return gate(x,mapping[task]) if gate is not None else x
    @torch.no_grad()
    def evaluate(expert,mode="learned"):
        if expert is not None:expert.eval()
        meter=PerformanceMeter(p,tasks);images=[];edge_error=0.
        for i in dev_indices[:2] if args.smoke else dev_indices:
            if time.time()>=deadline:emit('deadline_reached_during_evaluation');sys.exit(2)
            entry=dev[i];output={}
            sm,dm,nm,em=[meter.meters[t] for t in tasks]
            before=[np.asarray(v,dtype=float).copy() for v in (sm.tp,sm.fp,sm.fn)]
            prev=[float(dm.total_rmses),float(dm.n_valid),float(nm.sum_deg_diff),float(nm.total),float(em.loss),float(em.n)]
            for task in tasks:
                t=mapping[task];x=feature(entry,task)
                if expert is None:
                    logits=F.interpolate(heads[task](x),entry['batch']['image'].shape[-2:],mode='bilinear',align_corners=False)
                else:
                    logits=expert.forward_task(x,t,entry['context'][:,t:t+1].cuda(),entry['batch']['image'].shape[-2:],mode=mode,oracle_label=entry['batch'][task].cuda() if mode=='oracle' and task!='edge' else None,depth_min=float(p.TASKS.depth_min),depth_max=float(p.TASKS.depth_max))
                output[task]=get_output(logits,task)
            meter.update(output,{t:entry['batch'][t].cuda() for t in tasks})
            now=[float(dm.total_rmses),float(dm.n_valid),float(nm.sum_deg_diff),float(nm.total),float(em.loss),float(em.n)]
            images.append(dict(image=entry['image'],room=scene[entry['image']],
                tp=(np.asarray(sm.tp)-before[0]).tolist(),fp=(np.asarray(sm.fp)-before[1]).tolist(),fn=(np.asarray(sm.fn)-before[2]).tolist(),
                depth_sse=now[0]-prev[0],depth_n=now[1]-prev[1],normals_sum=now[2]-prev[2],normals_n=now[3]-prev[3],edge_sum=now[4]-prev[4],edge_n=now[5]-prev[5]))
        sm,dm,nm,em=[meter.meters[t] for t in tasks]
        tp,fp,fn=[np.asarray(v,float) for v in (sm.tp,sm.fp,sm.fn)]
        return dict(metrics=dict(mIoU=float(100*np.mean(tp/np.maximum(tp+fp+fn,1e-8))),rmse=float(np.sqrt(dm.total_rmses/dm.n_valid)),
            normals_deg=float(nm.sum_deg_diff/nm.total),edge_loss=float(em.loss/em.n)),per_image=images,edge_identity_max_abs=None)
    baseline=evaluate(None);write(d/'base_metrics.json',baseline)
    pool={}
    def teacher(family,seed):
        key=(family,seed)
        if key not in pool:
            folder=ART/'CVPR_sprint_20261024/mature_attribution'/('base'+str(base));path=folder/(family+'_seed'+str(seed)+'.pt')
            assert (folder/('result_'+family+'_seed'+str(seed)+'.json')).exists()
            z=torch.load(path,map_location='cpu');assert z['step']==3000
            ss=z['spec'];expert=make_expert(tasks,kind=ss['kind'],encoding=ss['encoding'],width=ss.get('width',128)).cuda()
            expert.load_state_dict(z['model'],strict=True);expert.eval().requires_grad_(False);pool[key]=(expert,str(path));del z
        return pool[key]
    for seed in args.seeds:
        rng=np.random.RandomState(seed+4000);schedule=[]
        while len(schedule)<args.steps:schedule.extend(rng.permutation(train_indices).tolist())
        schedule=schedule[:args.steps]
        for name in args.experts:
            result=d/('result_'+name+'_seed'+str(seed)+'.json')
            if result.exists():continue
            spec=VARIANTS[name];a,pa=teacher('factor_auto' if spec['family']=='event' else 'factor_charge_auto',seed)
            torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
            network=HeadTune(a,heads,tasks,spec).cuda()
            fit=spec['head'] or spec['calibration'];params=[v for v in network.parameters() if v.requires_grad] if fit else []
            checkpoint=d/(name+'_seed'+str(seed)+'.pt');optimizer=None;scheduler=None;history=[];start=0
            if fit:
                optimizer=torch.optim.Adam([{'params':[v for v in network.heads.parameters() if v.requires_grad],'lr':1e-4},{'params':[v for v in network.calibrator.parameters() if v.requires_grad],'lr':.003}],weight_decay=0.)
                scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda n:min(1.,(n+1)/min(10,args.steps))*max(0.,1-n/args.steps)**.9)
                if checkpoint.exists():
                    z=torch.load(checkpoint,map_location='cpu');assert z['spec']==spec and z['source_checkpoint']==pa and z['schedule']==schedule
                    network.restore(z['learned']);optimizer.load_state_dict(z['optimizer']);scheduler.load_state_dict(z['scheduler']);history=z['losses'];start=z['step'];restore_rng(z['rng']);del z
            began=time.time()
            def save(step):
                if not fit:return
                z=dict(learned=network.learned_state(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),step=step,losses=history,spec=spec,source_checkpoint=pa,tasks=tasks,depth_gain=1.25,base_seed=base,seed=seed,schedule=schedule,rng=snapshot_rng())
                temp=checkpoint.with_name(checkpoint.name+'.tmp');torch.save(z,temp);temp.replace(checkpoint)
            if fit:
                import signal
                def interrupted(sig,frame):save(len(history));emit('interrupted_checkpoint_saved',base=base,variant=name,step=len(history));sys.exit(143)
                signal.signal(signal.SIGTERM,interrupted)
                for step in range(start,args.steps):
                    if time.time()>=deadline:save(step);emit('deadline_reached');sys.exit(2)
                    network.train();optimizer.zero_grad(set_to_none=True);entry=train[schedule[step]];total=0.
                    for task in tasks:
                        if not network.trainable_for(task):continue
                        t=mapping[task];x=feature(entry,task);target=entry['batch'][task].cuda()
                        out=network.forward_task(x,t,entry['context'][:,t:t+1].cuda(),entry['batch']['image'].shape[-2:])
                        if task=='edge':loss=criterion({task:out},{task:target},tasks=[task])['total']
                        else:loss=metric_loss(out,target,task,float(p.TASKS.depth_min),float(p.TASKS.depth_max))
                        loss=loss+.01*network.regularizer(task)
                        assert torch.isfinite(loss);loss.backward();total+=float(loss.detach());del loss,out,x,target
                    for task in tasks:
                        ps=network.trainable_for(task)
                        if ps:
                            norm=torch.nn.utils.clip_grad_norm_(ps,5.);assert torch.isfinite(norm)
                    optimizer.step();scheduler.step();history.append(total)
                    if (step+1)%30==0 or step+1==args.steps:
                        save(step+1);emit('training',base=base,seed=seed,variant=name,step=step+1,target=args.steps,seconds=time.time()-began,peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)
            mode=spec.get('evaluation','learned');measured=evaluate(network,mode)
            diagnostic=mode=='oracle'
            row=dict(base_seed=base,seed=seed,variant=name,stage=mode,steps=args.steps if fit else 0,training_performed=fit,diagnostic_only=diagnostic,spec=spec,source_checkpoint=pa,checkpoint=str(checkpoint) if fit else None,trainable_parameters=sum(v.numel() for v in params),frozen_expert_parameters=sum(v.numel() for v in a.parameters()),depth_gain=1.25,seconds=time.time()-began,last_layer_lr=1e-4,calibration_lr=.003,anchor_strength=.01,train_images=96,dev_rooms=32,edge_head_trainable=spec['head'],**measured)
            write(result,row);emit('evaluated',base=base,seed=seed,variant=name,diagnostic_only=diagnostic,metrics=row['metrics'])
            del network,optimizer,scheduler;gc.collect();torch.cuda.empty_cache()
    del pool,heads,gate,criterion;gc.collect();torch.cuda.empty_cache()
emit('completed',bases=args.bases,seeds=args.seeds,steps=args.steps)
