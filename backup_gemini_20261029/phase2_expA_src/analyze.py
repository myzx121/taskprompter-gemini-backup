import pathlib,json,statistics,datetime,csv
R=pathlib.Path(__file__).resolve().parent;rows=[]
for p in sorted((R/'head120').glob('base*/result_*.json')):
 d=json.loads(p.read_text());d['source']=str(p.relative_to(R));rows.append(d)
keys=['mIoU','rmse','normals_deg','edge_loss'];ix={(d['base_seed'],d['seed'],d['variant']):d for d in rows}
lines=['# 预测头末层解冻与校准实验','',str(datetime.datetime.utcnow())+' UTC',f'完成 {len(rows)}/32，其中120步更新 {sum(d["training_performed"] for d in rows)}/24。','',
 '|基座|种子|配置|步数|mIoU|RMSE|法线°|edge loss|','|---|---|---|---|---|---|---|---|']
for d in rows:lines.append('|'+ '|'.join(map(str,[d['base_seed'],d['seed'],d['variant'],d['steps']]+[d['metrics'][k] for k in keys]))+'|')
lines+=['','## 相对同源冻结参照的配对均值','','|基座|来源|配置|n|ΔmIoU pp|ΔRMSE|Δ法线°|Δedge loss|','|---|---|---|---|---|---|---|---|']
for base in (2038,2039):
 for family in ('event','charge'):
  for mode in ('calibration','head','head_calibration'):
   pairs=[(ix[base,s,family+'_'+mode],ix[base,s,family+'_none']) for s in (0,1) if (base,s,family+'_'+mode) in ix and (base,s,family+'_none') in ix]
   if pairs:lines.append('|'+ '|'.join(map(str,[base,family,mode,len(pairs)]+[statistics.mean(a['metrics'][k]-b['metrics'][k] for a,b in pairs) for k in keys]))+'|')
lines+=['','## 组合交互：组合−末层解冻−校准＋冻结参照','','|基座|来源|n|交互mIoU pp|交互RMSE|交互法线°|交互edge loss|','|---|---|---|---|---|---|---|']
for base in (2038,2039):
 for family in ('event','charge'):
  names=[family+'_'+m for m in ('head_calibration','head','calibration','none')];seeds=[s for s in (0,1) if all((base,s,n) in ix for n in names)]
  if seeds:lines.append('|'+ '|'.join(map(str,[base,family,len(seeds)]+[statistics.mean(sum(c*ix[base,s,n]['metrics'][k] for c,n in zip((1,-1,-1,1),names)) for s in seeds) for k in keys]))+'|')
lines+=['','当前不更新Task Prompts和骨干；Edge只更新原预测头末层，没有脉冲注入。edge loss不是ODS。固定样本小训练不能证明全量收敛或SOTA；未完成结果不填零。']
(R/'PROGRESS.md').write_text('\n'.join(lines),encoding='utf-8')
with (R/'head_metrics.csv').open('w',newline='',encoding='utf-8') as f:
 fields=['source','base_seed','seed','variant','steps','training_performed','trainable_parameters']+keys;w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for d in rows:w.writerow({**{k:d[k] for k in fields if k not in keys},**{k:d['metrics'][k] for k in keys}})
print('completed',len(rows))
