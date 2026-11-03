"""Read-only replay adapter. Python standard library only; no automatic workface detection."""
import argparse, csv, hashlib, json, pathlib, math

def build(folder, output, measurements=None):
    folder, output = pathlib.Path(folder), pathlib.Path(output)
    names = ['classification.csv', 'feature_lines.csv', 'grid_origin.txt']
    hashes = {n: hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in names}
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    ox, oy, cell = map(float, (folder/'grid_origin.txt').read_text().split())
    lines = {}
    with (folder/'feature_lines.csv').open(newline='', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            i, t = int(r['line_id']), int(r['type'])
            l = lines.setdefault(i, dict(id=i, type=t, p=[]))
            if t != l['type'] or int(r['point_order']) != len(l['p']):
                raise ValueError('Non-contiguous line order/type')
            l['p'].append([float(r['x'])-ox, float(r['y'])-oy, float(r['z'])])
    for l in lines.values():
        l['s'] = [0]
        for a,b in zip(l['p'],l['p'][1:]): l['s'].append(l['s'][-1]+math.hypot(b[0]-a[0],b[1]-a[1]))
        l['hash'] = hashlib.sha256(json.dumps(l,sort_keys=True).encode()).hexdigest()
    grid=[]; rows=cols=0
    with (folder/'classification.csv').open(newline='',encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            y,x=int(r['r']),int(r['c']); rows=max(rows,y+1); cols=max(cols,x+1)
            if int(r['count'])>0: grid.append([y,x,float(r['z']),int(r['type'])])
    data=dict(id=identity,hashes=hashes,origin=[ox,oy],cell=cell,rows=rows,cols=cols,lines=list(lines.values()),grid=grid,
              name=folder.name,baseline='9d3d2cc34f31273296e487b9cd7e47b63f4d51bc')
    # Legacy measurements lack a verified dataset manifest: retained as references only.
    data['legacy']=[]
    data['measurements']=[]
    for name in ['matched_sections.csv','platform_widths.csv']:
        if (folder/name).exists():
            data['legacy'].append(dict(file=name,sha256=hashlib.sha256((folder/name).read_bytes()).hexdigest(),status='dataset_and_ownership_unverified'))
    if measurements:
        for name,kind in [('matched_sections.csv','bench'),('platform_widths.csv','platform')]:
            src=pathlib.Path(measurements)/name
            if not src.exists(): continue
            sha=hashlib.sha256(src.read_bytes()).hexdigest()
            with src.open(newline='',encoding='utf-8-sig') as f:
                for row in csv.DictReader(f):
                    if 'crest_line_id' not in row: continue
                    data['measurements'].append(dict(kind=kind,source=name,sourceHash=sha,row=row))
    here=pathlib.Path(__file__).parent
    html=(here/'viewer.html').read_text(encoding='utf-8').replace('/*__ENGINE__*/',(here/'engine.js').read_text(encoding='utf-8')).replace('/*__DATA__*/','const DATA='+json.dumps(data,separators=(',',':'),allow_nan=False).replace('<','\\u003c')+';').replace('/*__UI__*/',(here/'ui.js').read_text(encoding='utf-8'))
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(html,encoding='utf-8')
    (output.parent/'dataset-manifest.json').write_text(json.dumps({k:v for k,v in data.items() if k not in ['lines','grid','measurements']},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(output),dataset=identity,lines=len(lines),observed_cells=len(grid),rows=rows,cols=cols)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('output');p.add_argument('--measurements');a=p.parse_args();build(a.input,a.output,a.measurements)
