import pathlib,sys,subprocess,json,datetime
R=pathlib.Path(__file__).resolve().parent
if '--worker' in sys.argv:
 with (R/'head_tuning.log').open('a') as f:p=subprocess.run([sys.executable,str(R/'run_head_tuning.py')],cwd=str(R),stdout=f,stderr=subprocess.STDOUT)
 phase='completed' if p.returncode==0 else ('interrupted' if p.returncode in (143,-15) else 'failed')
 (R/'worker.json').write_text(json.dumps(dict(phase=phase,returncode=p.returncode,utc=datetime.datetime.utcnow().isoformat())))
 subprocess.run([sys.executable,str(R/'analyze.py')],cwd=str(R))
else:
 with (R/'worker.log').open('a') as f:p=subprocess.Popen([sys.executable,str(__file__),'--worker'],cwd=str(R),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 d=dict(pid=p.pid,phase='running',utc=datetime.datetime.utcnow().isoformat(),evaluations=32,gradient_fits=24,steps=120,train_images=96,dev_rooms=32,automatic_large_scale_promotion=False,deadline_beijing='2026-11-16 00:00')
 (R/'launch_receipt.json').write_text(json.dumps(d,indent=2));(R/'worker.json').write_text(json.dumps(d));print(json.dumps(d))
