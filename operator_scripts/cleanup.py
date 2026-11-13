import os, glob, shutil, subprocess, time
H = os.path.expanduser('~/NatureBench')
names = subprocess.run(['docker', 'ps', '--format', '{{.Names}}'], capture_output=True, text=True).stdout
running = lambda t: t in names
EXT = ('.pt','.pth','.ckpt','.bin','.npy','.npz','.h5','.hdf5','.mmap','.safetensors','.parquet','.pkl','.joblib','.arrow','.feather','.dat')
freed = 0
def size(p):
    try: return os.path.getsize(p)
    except OSError: return 0
def tree_size(d):
    return sum(size(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs)
for d in glob.glob(H + '/results/*/*/'):
    t = os.path.basename(d.rstrip('/'))
    if running(t) or not os.path.exists(d + 'result.json'): continue
    for r, _, fs in os.walk(d + 'workspace'):
        for f in fs:
            p = os.path.join(r, f)
            if f.lower().endswith(EXT) and size(p) > 100e6:
                freed += size(p); os.remove(p)
    snap = d + 'workspace/.optimus/outputs'
    if os.path.isdir(snap):
        freed += tree_size(snap); shutil.rmtree(snap, ignore_errors=True)
for d in glob.glob(H + '/data/naturebench_data/tasks/*/'):
    t = os.path.basename(d.rstrip('/'))
    if running(t) or not glob.glob(f'{H}/results/v11_*/{t}/result.json'): continue
    pdata = d + 'problem/data'
    if os.path.isdir(pdata):
        freed += tree_size(pdata); shutil.rmtree(pdata, ignore_errors=True)
free = shutil.disk_usage('/').free
print(f'[cleanup {time.strftime("%Y-%m-%d %H:%M")}] freed {freed/1e9:.1f} GB, {free/1e9:.0f} GB free', flush=True)
