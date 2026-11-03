"""Offline review and lossless provenance verification; standard library only."""
import csv, json, sys, pathlib, zlib, struct, base64
from collections import defaultdict, Counter
source, out = map(pathlib.Path, sys.argv[1:3])
def rows(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))
origin = list(map(float, (source/'grid_origin.txt').read_text().split()))
raw = rows(source/'feature_lines.csv')
lines = {}
original = {}
for r in raw:
    key = int(r['line_id'])
    p = [float(r[n]) for n in ('x','y','z')]
    original[key,int(r['point_order'])] = p
    lines.setdefault(key, {'id':key,'type':int(r['type']),'p':[]})['p'].append([p[0]-origin[0],p[1]-origin[1]])
fixed = rows(out/'feature_lines_repaired.csv')
provenance = rows(out/'feature_gap_provenance.csv')
assert len(fixed) == len(provenance) == len(raw), 'Point count changed'
seen = set()
for f,p in zip(fixed,provenance):
    assert f['line_id']==p['repaired_line_id'] and f['point_order']==p['point_order']
    key = int(p['source_line_id']),int(p['source_point_order'])
    assert key not in seen
    seen.add(key)
    # C++ input uses float; original CSV contains decimal rounded floats.
    expected=[struct.unpack('f',struct.pack('f',x))[0] for x in original[key]]
    assert all(abs(float(f[n])-v)<1e-5 for n,v in zip(('x','y','z'),expected))
assert seen==set(original)
candidates=rows(out/'feature_gap_candidates.csv')
assert sum(int(p['incoming_inferred_bridge_id'])>=0 for p in provenance)==sum(c['status']=='accepted' for c in candidates)
colors={0:(245,245,245),1:(148,198,152),2:(218,142,140),4:(255,219,67)}
grid=rows(source/'classification.csv')
w=max(int(r['c']) for r in grid)+1; h=max(int(r['r']) for r in grid)+1
pixels=bytearray([245])*(w*h*3)
for r in grid:
    i=((h-1-int(r['r']))*w+int(r['c']))*3
    pixels[i:i+3]=bytes(colors.get(int(r['type']),colors[0]))
def chunk(kind,data):
    return struct.pack('!I',len(data))+kind+data+struct.pack('!I',zlib.crc32(kind+data)&0xffffffff)
png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!2I5B',w,h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(b''.join(b'\0'+pixels[y*w*3:(y+1)*w*3] for y in range(h))))+chunk(b'IEND',b'')
data=dict(lines=list(lines.values()),candidates=candidates,origin=origin,w=w,h=h)
html='''<!doctype html><meta charset="utf-8"><title>特征线短断口核验</title>
<style>body{font:15px system-ui;margin:16px;background:#f5f7fa}header{padding:12px;background:white}canvas{width:100%;height:68vh;background:white}button,select{padding:8px;margin:5px}#info{white-space:pre-wrap}</style>
<header><h2>特征线短断口修复 · 人工核验</h2><p>黑：原坡顶；蓝：原坡底；洋红粗线：推断补线。补线未进入正式参数计算。滚轮缩放、拖动平移。</p>
<button id="fit">全场</button><label><input id="after" type="checkbox" checked>显示补线</label>
<label><input id="ends" type="checkbox">显示全部端点</label><select id="pick"><option value="">选择候选缺口</option></select><div id="info"></div></header><canvas id="map"></canvas>
<script>const D=__DATA__;const im=new Image();im.src='data:image/png;base64,__PNG__';
const c=document.querySelector('#map'),ctx=c.getContext('2d'),pick=document.querySelector('#pick'),after=document.querySelector('#after'),ends=document.querySelector('#ends'),info=document.querySelector('#info');
let view=[D.w*D.origin[2]/2,D.h*D.origin[2]/2,D.w*D.origin[2]*1.03],drag=null;
const accepted=D.candidates.filter(x=>x.status==='accepted');
for(const r of [...D.candidates].sort((a,b)=>(a.status==='accepted'?-1:1)-(b.status==='accepted'?-1:1)||+a.candidate_id-+b.candidate_id)){let o=document.createElement('option');o.value=r.candidate_id;o.textContent=`#${r.candidate_id} ${r.status}  ${r.line_a} ↔ ${r.line_b}`;pick.append(o)}
function xy(p){let s=c.width/view[2];return [(p[0]-view[0])*s+c.width/2,c.height/2-(p[1]-view[1])*s]}
function path(p,color,width){ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();p.forEach((v,i)=>{let q=xy(v);i?ctx.lineTo(...q):ctx.moveTo(...q)});ctx.stroke()}
function draw(){c.width=c.clientWidth*devicePixelRatio;c.height=c.clientHeight*devicePixelRatio;ctx.clearRect(0,0,c.width,c.height);let s=c.width/view[2],q=xy([0,D.h*D.origin[2]]);ctx.imageSmoothingEnabled=false;ctx.drawImage(im,q[0],q[1],D.w*D.origin[2]*s,D.h*D.origin[2]*s);
for(let l of D.lines){path(l.p,l.type?'#1757d8':'#111',devicePixelRatio);if(ends.checked)for(let p of [l.p[0],l.p.at(-1)]){ctx.fillStyle='#ff7800';ctx.beginPath();ctx.arc(...xy(p),3*devicePixelRatio,0,7);ctx.fill()}}
if(after.checked)for(let r of accepted)path([[+r.ax-D.origin[0],+r.ay-D.origin[1]],[+r.bx-D.origin[0],+r.by-D.origin[1]]],'#d000e8',4*devicePixelRatio);
let r=D.candidates.find(x=>x.candidate_id===pick.value);if(r){ctx.strokeStyle='#ff6800';for(let p of [[+r.ax-D.origin[0],+r.ay-D.origin[1]],[+r.bx-D.origin[0],+r.by-D.origin[1]]]){ctx.beginPath();ctx.arc(...xy(p),7*devicePixelRatio,0,7);ctx.stroke()}}
}
pick.onchange=()=>{let r=D.candidates.find(x=>x.candidate_id===pick.value);if(!r)return;view=[(+r.ax+ +r.bx)/2-D.origin[0],(+r.ay+ +r.by)/2-D.origin[1],35];info.textContent=`候选 ${r.candidate_id} | ${r.status} | 原线 ${r.line_a} 端 ${r.end_a} → ${r.line_b} 端 ${r.end_b}\n间距 ${(+r.gap_xy).toFixed(2)} m；高程趋势残差 ${(+r.z_residual).toFixed(2)} m；方向偏差 ${(+r.angle).toFixed(1)}°（早期拒绝时未计算项为0）`;draw()};
document.querySelector('#fit').onclick=()=>{view=[D.w*D.origin[2]/2,D.h*D.origin[2]/2,D.w*D.origin[2]*1.03];draw()};after.onchange=ends.onchange=draw;
c.onwheel=e=>{e.preventDefault();view[2]*=e.deltaY>0?1.2:1/1.2;view[2]=Math.max(5,Math.min(5000,view[2]));draw()};c.onpointerdown=e=>{drag=[e.clientX,e.clientY,...view];c.setPointerCapture(e.pointerId)};c.onpointerup=()=>drag=null;c.onpointermove=e=>{if(drag){view[0]=drag[2]-(e.clientX-drag[0])*view[2]/c.clientWidth;view[1]=drag[3]+(e.clientY-drag[1])*view[2]/c.clientWidth;draw()}};window.onresize=draw;im.onload=draw;
info.textContent=`共 ${D.lines.length} 条原线；${D.candidates.length} 个半径内同类候选；接受 ${accepted.length} 条补线。`;
</script>'''.replace('__DATA__',json.dumps(data)).replace('__PNG__',base64.b64encode(png).decode())
(out/'feature_gap_review.html').write_text(html,encoding='utf-8')
print('Verified every original vertex exactly once:',len(raw))
print('Original / repaired lines:',len(lines),len(set(r['line_id'] for r in fixed)))
print(Counter(r['status'] for r in candidates))
