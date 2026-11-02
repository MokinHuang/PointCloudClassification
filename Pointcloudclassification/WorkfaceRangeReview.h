#pragma once
#include "AutoWorkfaces.h"
#include <cstdint>
#include <sstream>

namespace range_review {
// Dataset identity is for accidental cross-run mixups, NOT a security signature.
inline std::string fingerprint(const std::vector<FeatureLine>& lines,const std::vector<std::vector<GridCell>>& grid,
                              double cell,const Eigen::Vector4f& origin){
    uint64_t h=14695981039346656037ULL;
    auto bytes=[&](const auto& value){const auto* b=reinterpret_cast<const unsigned char*>(&value);for(size_t i=0;i<sizeof(value);++i){h^=b[i];h*=1099511628211ULL;}};
    bytes(cell);for(int i=0;i<4;++i)bytes(origin[i]);bytes(lines.size());
    for(auto& l:lines){bytes(l.id);bytes(l.type);bytes(l.points.size());for(auto& p:l.points)for(int k=0;k<3;++k)bytes(p[k]);}
    bytes(grid.size());for(auto& row:grid){bytes(row.size());for(auto& g:row){bytes(g.avg_z);bytes(g.count);bytes(g.type);bytes(g.is_empty);bytes(g.is_crest);bytes(g.is_toe);}}
    std::ostringstream s;s<<std::hex<<h;return s.str();
}
inline void write(const std::vector<FeatureLine>& source,const std::vector<std::vector<GridCell>>& grid,double cell,
                  const Eigen::Vector4f& origin,const autowalls::Result& graph,const std::vector<overall::Preview>& previews,const std::string& prefix){
    if(grid.empty()||grid.front().empty())throw std::runtime_error("Empty range review grid");
    auto lines=overall::localLines(source,origin);
    std::ofstream out(prefix+"workface_range_review.html");if(!out)throw std::runtime_error("Cannot write range review");
    out<<R"HTML(<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>工作帮范围确认</title><style>
*{box-sizing:border-box}body{font:15px system-ui,sans-serif;background:#f2f5f8;color:#17283e;margin:16px}h1{font-size:23px;margin:4px 0}p{margin:10px 0;line-height:1.6}.card{background:white;border:1px solid #d5dee8;border-radius:8px;padding:14px;margin:10px 0}label{display:inline-flex;align-items:center;gap:5px;margin:5px 8px 5px 0}button,input,select{font:inherit;padding:6px;max-width:100%}button{cursor:pointer;margin:4px;border:1px solid #a6b6c9;border-radius:4px;background:#f5f8fc}input[type=number]{width:100px}input[type=checkbox]{width:auto}input[type=text]{width:180px}#map{width:100%;height:65vh;display:block;background:#fff;touch-action:none;border:1px solid #ccc}#status{white-space:pre-wrap;color:#8b3c00}#state{padding:5px 12px;background:#fff0c2;border-radius:5px}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:7px;border-bottom:1px solid #e1e6ec}.scroll{max-height:240px;overflow:auto}.small{font-size:13px;color:#526479}.checks{display:grid;gap:4px}.checks label{align-items:flex-start}#scopeInfo{white-space:pre-wrap}.confirmed{background:#d3f2dd!important;color:#126337}
</style>
<h1>工作帮范围确认 <span id="state">未选择</span></h1>
<p>按论文图5的几何定义：完整多台阶工作帮的最上部坡顶 → 最下部坡底。这里只确认范围；不自动证明采剥状态、参数精度或安全性。</p>
<div class="card"><label>自动候选<select id="candidate"><option value="">请选择候选</option></select></label><button id="new">新建范围</button><label>已保存范围<select id="saved"><option value="">请选择记录</option></select></label><button id="delete">删除当前记录</button>
<div><label>范围编号<input id="name" type="text" value="WF_1"></label><label>核验人<input id="reviewer" type="text" maxlength="80" placeholder="输入姓名或代号"></label><label>备注<input id="note" type="text" maxlength="500" style="width:360px"></label></div>
<div><label>上部坡顶<select id="upper"><option value="">请选择</option></select></label><label>起点/m<input id="uStart" type="number" value="0" step="0.1"></label><label>终点/m<input id="uEnd" type="number" value="0" step="0.1"></label></div>
<div><label>下部坡底<select id="lower"><option value="">请选择</option></select></label><label>起点/m<input id="lStart" type="number" value="0" step="0.1"></label><label>终点/m<input id="lEnd" type="number" value="0" step="0.1"></label></div>
<p class="small">区段距离均沿对应特征线的原始顶点顺序累计。上下线方向可能相反；只能在各自线长内填写起点&lt;终点。橙色：选定上部坡顶；青色：选定下部坡底；黄色：中间台阶边界参考线。高亮整条中间线不代表整条线都属于本范围。</p>
<button id="find">查找两边界之间的台阶链</button><label>可用路径<select id="paths"><option value="">请选择路径</option></select></label><button id="focus">定位当前范围</button><button id="fit">全场</button>
<label>地图点击<select id="pick"><option value="inspect">查看线编号</option><option value="upper">选择上部坡顶</option><option value="lower">选择下部坡底</option><option value="uStart">上部区段起点</option><option value="uEnd">上部区段终点</option><option value="lStart">下部区段起点</option><option value="lEnd">下部区段终点</option></select></label>
<label><input id="showSections" type="checkbox" checked>显示原候选的可用参考测线</label></div>
<canvas id="map" aria-label="工作帮上下边界和中间台阶"></canvas>
<div class="card"><h3>从上到下的台阶链</h3><p id="scopeInfo"></p><div class="scroll"><table><thead><tr><th>顺序</th><th>节点</th><th>坡顶 → 坡底</th><th>平均高程/m</th><th>到下一台阶的平台支持</th><th>调整</th></tr></thead><tbody id="chain"></tbody></table></div>
<label>追加台阶节点<select id="node"></select></label><button id="append">追加</button><p class="small">节点来自已有合格单台阶配对；支持不足或断开的台阶链可保存草稿，但不能确认。自动图的末端不一定是真实工作帮末端，须结合全场和实际资料核验。</p>
<div class="checks"><label><input id="checkTop" type="checkbox">我已核实上边界是此范围最上部台阶的坡顶，不是中途台阶。</label><label><input id="checkBottom" type="checkbox">我已核实下边界是此范围最下部台阶的坡底，不是中途台阶。</label><label><input id="checkChain" type="checkbox">我已核实区段及中间台阶归属，不跨另一侧边坡；没有把裁切或缺测误认作完整边界。</label></div>
<button id="saveDraft">保存草稿</button><button id="confirm">确认并保存此范围</button><button id="revoke">撤销确认</button><p id="status" role="status" aria-live="polite">请选择候选后核验。改变任一范围字段将撤销确认。</p></div>
<div class="card"><button id="export">下载范围记录 JSON</button><label>恢复 JSON<input id="import" type="file" accept=".json,application/json"></label><button id="localSave">备份到本浏览器</button><button id="localLoad">恢复浏览器备份</button><button id="preview">导出兼容预览 CSV</button>
<p class="small">JSON 是完整范围记录，含上下区段、台阶链、人工核验状态及本次数据指纹；重新提取后不得混用。浏览器不会自动写入项目目录，请下载备份。CSV 只用于旧版几何预览，不包含下边界区段限制和人工确认状态，不是正式测量配置。</p></div>
<script>
const D=)HTML";
    out<<std::setprecision(17)<<"{\"dataset\":\""<<fingerprint(source,grid,cell,origin)<<"\",\"cell\":"<<cell<<",\"rows\":"<<grid.size()<<",\"cols\":"<<grid[0].size()<<",\"lines\":[";
    bool first=true;for(auto& l:lines){if(!first)out<<',';first=false;out<<"{\"id\":"<<l.id<<",\"type\":"<<l.type<<",\"length\":"<<l.s.back()<<",\"p\":[";
        for(size_t k=0;k<l.p.size();++k){if(k)out<<',';out<<'['<<l.p[k].x()<<','<<l.p[k].y()<<','<<l.p[k].z()<<','<<l.s[k]<<']';}out<<"]}";}
    out<<"],\"runs\":[";first=true;
    for(size_t r=0;r<grid.size();++r)for(size_t c=0;c<grid[r].size();){int type=grid[r][c].count>0?grid[r][c].type:-1;size_t e=c+1;while(e<grid[r].size()&&(grid[r][e].count>0?grid[r][e].type:-1)==type)++e;
        if(!first)out<<',';first=false;out<<'['<<r<<','<<c<<','<<e-c<<','<<type<<']';c=e;}
    out<<"],\"nodes\":[";
    for(size_t i=0;i<graph.nodes.size();++i){auto& n=graph.nodes[i];if(i)out<<',';out<<"{\"id\":"<<i<<",\"crest\":"<<n.crest<<",\"toe\":"<<n.toe<<",\"high\":"<<n.high<<",\"low\":"<<n.low<<",\"support\":"<<n.samples.size()<<'}';}
    out<<"],\"edges\":[";for(size_t i=0;i<graph.edges.size();++i){auto e=graph.edges[i];if(i)out<<',';out<<'['<<e.upper<<','<<e.lower<<','<<e.support<<']';}
    out<<"],\"candidates\":[";for(size_t i=0;i<graph.candidates.size();++i){auto& c=graph.candidates[i];if(i)out<<',';
        out<<"{\"name\":\""<<c.selection.name<<"\",\"upper\":"<<c.selection.upper<<",\"lower\":"<<c.selection.lower<<",\"start\":"<<c.selection.start<<",\"end\":"<<c.selection.end<<",\"chain\":[";
        for(size_t k=0;k<c.chain.size();++k){if(k)out<<',';out<<c.chain[k];}out<<"]}";}
    out<<"],\"sections\":[";first=true;for(auto& s:previews)if(s.geometryAvailable){if(!first)out<<',';first=false;
        out<<"{\"name\":\""<<s.name<<"\",\"station\":"<<s.station<<",\"a\":["<<s.top.x()-origin.x()<<','<<s.top.y()-origin.y()<<"],\"b\":["<<s.bottom.x()-origin.x()<<','<<s.bottom.y()-origin.y()<<"]}";}
    out<<R"HTML(]};
const $=id=>document.getElementById(id),canvas=$('map'),ctx=canvas.getContext('2d');
const lineById=new Map(D.lines.map(x=>[x.id,x])),nodeById=new Map(D.nodes.map(x=>[x.id,x]));
const edgeByKey=new Map(D.edges.map(e=>[e[0]+':'+e[1],e[2]]));
let chain=[],sourceCandidate='',state='draft',confirmedAt=null,activeSaved=null,records=new Map(),paths=[],drag=null;
let view=[D.cols*D.cell/2,D.rows*D.cell/2,D.cols*D.cell*1.04];
const copy=x=>JSON.parse(JSON.stringify(x));
function option(select,value,label){const o=document.createElement('option');o.value=value;o.textContent=label;select.append(o);}
for(const l of D.lines)option($(l.type===0?'upper':'lower'),l.id,`${l.id} / ${l.type===0?'坡顶':'坡底'} (${l.length.toFixed(1)} m)`);
for(const n of D.nodes)option($('node'),n.id,`${n.id}: 坡顶 ${n.crest} → 坡底 ${n.toe}`);
const validCounts=new Map();for(const s of D.sections)validCounts.set(s.name,(validCounts.get(s.name)||0)+1);
for(const c of [...D.candidates].sort((a,b)=>(validCounts.get(b.name)||0)-(validCounts.get(a.name)||0)))option($('candidate'),c.name,`${c.name} / ${c.chain.length}级候选 / ${validCounts.get(c.name)||0}条参考测线`);
function message(s){$('status').textContent=s;}
function current(){return {id:$('name').value.trim(),source_candidate:sourceCandidate,upper:Number($('upper').value||-1),lower:Number($('lower').value||-1),upper_start:Number($('uStart').value),upper_end:Number($('uEnd').value),lower_start:Number($('lStart').value),lower_end:Number($('lEnd').value),chain:[...chain],reviewer:$('reviewer').value.trim(),note:$('note').value.trim(),status:state,confirmed_at:confirmedAt,checks:{top:$('checkTop').checked,bottom:$('checkBottom').checked,chain:$('checkChain').checked}};}
function validate(r,confirmation=false){
 if(!/^[A-Za-z0-9_-]{1,64}$/.test(r.id))return '编号须为1–64位英文字母、数字、下划线或短横线。';
 const up=lineById.get(r.upper),lo=lineById.get(r.lower);
 if(!up||!lo||up.type!==0||lo.type!==1)return '必须选择上部坡顶和下部坡底。';
 for(const [a,b,l] of [[r.upper_start,r.upper_end,up],[r.lower_start,r.lower_end,lo]])if(!Number.isFinite(a)||!Number.isFinite(b)||a<0||b<=a||b>l.length+1e-7)return '区段必须满足 0 ≤ 起点 < 终点 ≤ 对应线长。';
 if(!Array.isArray(r.chain)||r.chain.some(id=>!Number.isInteger(id)||!nodeById.has(id))||new Set(r.chain).size!==r.chain.length)return '台阶链包含未知或重复节点。';
 if(typeof r.reviewer!=='string'||r.reviewer.length>80||typeof r.note!=='string'||r.note.length>500)return '核验人或备注格式无效。';
 if(!confirmation)return '';
 if(r.chain.length<2)return '工作帮至少需要两个台阶节点；请完善台阶链。';
 if(nodeById.get(r.chain[0]).crest!==r.upper||nodeById.get(r.chain.at(-1)).toe!==r.lower)return '首尾台阶与选择的上部坡顶、下部坡底不一致。';
 for(let i=1;i<r.chain.length;i++)if(!edgeByKey.has(r.chain[i-1]+':'+r.chain[i]))return '台阶之间缺少可靠平台连接；可以保存草稿，不能确认。';
 if(!r.reviewer.trim())return '请填写核验人。';
 if(!r.checks||r.checks.top!==true||r.checks.bottom!==true||r.checks.chain!==true)return '请逐项核实并勾选三个范围确认项。';
 return '';
}
function refreshSaved(){const value=activeSaved;$('saved').replaceChildren();option($('saved'),'','请选择记录');for(const r of records.values())option($('saved'),r.id,`${r.id} / ${r.status==='range_confirmed'?'范围已确认':'草稿'}`);$('saved').value=value||'';}
function clearChecks(){for(const id of ['checkTop','checkBottom','checkChain'])$(id).checked=false;}
function invalidate(text='范围已更改，原确认已撤销；请重新核验。'){
 state='draft';confirmedAt=null;clearChecks();
 if(activeSaved&&records.has(activeSaved)){const old=records.get(activeSaved);old.status='draft';old.confirmed_at=null;old.checks={top:false,bottom:false,chain:false};refreshSaved();}
 message(text);render();
}
function station(l,p){let best=Infinity,result=0;for(let i=1;i<l.p.length;i++){const a=l.p[i-1],b=l.p[i],dx=b[0]-a[0],dy=b[1]-a[1],n=dx*dx+dy*dy,t=n?Math.max(0,Math.min(1,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/n)):0,d=Math.hypot(p[0]-a[0]-t*dx,p[1]-a[1]-t*dy);if(d<best){best=d;result=a[3]+t*(b[3]-a[3]);}}return {s:result,d:best};}
function interpolate(l,s){for(let i=1;i<l.p.length;i++)if(l.p[i][3]>=s){const a=l.p[i-1],b=l.p[i],t=(s-a[3])/(b[3]-a[3]);return [a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1])];}return l.p.at(-1).slice(0,2);}
function clipped(l,a,b){if(!l||!Number.isFinite(a)||!Number.isFinite(b)||a<0||b<=a||b>l.length+1e-7)return [];return [interpolate(l,a),...l.p.filter(p=>p[3]>a&&p[3]<b),interpolate(l,b)];}
function relevantSections(){const r=current(),candidate=D.candidates.find(c=>c.name===sourceCandidate);if(!candidate||candidate.upper!==r.upper||candidate.lower!==r.lower||candidate.chain.join(',')!==chain.join(','))return [];
 const lo=lineById.get(r.lower);return D.sections.filter(s=>s.name===sourceCandidate&&s.station>=r.upper_start-1e-6&&s.station<=r.upper_end+1e-6&&station(lo,s.b).s>=r.lower_start-1e-6&&station(lo,s.b).s<=r.lower_end+1e-6);}
function setForm(r){$('name').value=r.id;$('upper').value=r.upper;$('lower').value=r.lower;for(const [id,key] of [['uStart','upper_start'],['uEnd','upper_end'],['lStart','lower_start'],['lEnd','lower_end']])$(id).value=r[key];$('reviewer').value=r.reviewer;$('note').value=r.note;chain=[...r.chain];sourceCandidate=r.source_candidate||'';state=r.status;confirmedAt=r.confirmed_at;for(const [id,key] of [['checkTop','top'],['checkBottom','bottom'],['checkChain','chain']])$(id).checked=!!r.checks?.[key];$('candidate').value=sourceCandidate;paths=[];$('paths').replaceChildren();option($('paths'),'','请选择路径');render();}
function loadCandidate(){const c=D.candidates.find(x=>x.name===$('candidate').value);if(!c)return;
 const up=lineById.get(c.upper),lo=lineById.get(c.lower);activeSaved=null;
 setForm({id:'WF_'+c.name,upper:c.upper,lower:c.lower,upper_start:c.start,upper_end:Math.min(c.end,up.length),lower_start:0,lower_end:lo.length,chain:c.chain,source_candidate:c.name,reviewer:$('reviewer').value.trim(),note:'',status:'draft',confirmed_at:null,checks:{}});
 refreshSaved();message('已加载自动候选，默认下部范围为整条线；请调整区段并核实是否为完整工作帮。');focus();}
)HTML";
    out<<R"HTML(
function table(){const body=$('chain');body.replaceChildren();chain.forEach((id,i)=>{const n=nodeById.get(id),tr=document.createElement('tr');if(!n)return;
 const support=i+1<chain.length?edgeByKey.get(id+':'+chain[i+1]):null;
 for(const value of [i+1,id,`${n.crest} → ${n.toe}`,`${n.high.toFixed(2)} → ${n.low.toFixed(2)}`,i+1===chain.length?'末级':support?`${support} 条可靠测线`:'缺少连接']){const td=document.createElement('td');td.textContent=value;tr.append(td);}
 const td=document.createElement('td');for(const [label,action,disabled] of [['上移',()=>{[chain[i-1],chain[i]]=[chain[i],chain[i-1]];},i===0],['下移',()=>{[chain[i+1],chain[i]]=[chain[i],chain[i+1]];},i+1===chain.length],['移除',()=>chain.splice(i,1),false]]){const b=document.createElement('button');b.textContent=label;b.disabled=disabled;b.onclick=()=>{action();invalidate();};td.append(b);}tr.append(td);body.append(tr);});}
function render(){table();$('state').textContent=state==='range_confirmed'?'范围已确认（未测量认证）':'草稿 / 未确认';$('state').className=state==='range_confirmed'?'confirmed':'';
 const r=current(),err=validate(r),sections=relevantSections();$('scopeInfo').textContent=`上部坡顶 ${r.upper} [${r.upper_start.toFixed(2)}, ${r.upper_end.toFixed(2)}] m → 下部坡底 ${r.lower} [${r.lower_start.toFixed(2)}, ${r.lower_end.toFixed(2)}] m\n${chain.length} 个台阶节点；${sections.length} 条原候选参考测线落在当前区段。${err?'\n检查：'+err:''}\n调整边界或链后不会在浏览器重算截面；无参考测线不等于无坡面，后续仍需重新计算。`;draw();}
function screen(p){const s=canvas.width/view[2];return [(p[0]-view[0])*s+canvas.width/2,canvas.height/2-(p[1]-view[1])*s];}
function world(x,y){const s=canvas.clientWidth/view[2];return [(x-canvas.clientWidth/2)/s+view[0],view[1]-(y-canvas.clientHeight/2)/s];}
const background=document.createElement('canvas');background.width=D.cols;background.height=D.rows;const bg=background.getContext('2d'),palette={'-1':'#f1f1f1',0:'#ddd',1:'#a0cba3',2:'#e3a4a0',4:'#ffdb49'};
for(const [r,c,n,t] of D.runs){bg.fillStyle=palette[t]||'#ddd';bg.fillRect(c,D.rows-1-r,n,1);}
function draw(){const d=devicePixelRatio||1;canvas.width=Math.round(canvas.clientWidth*d);canvas.height=Math.round(canvas.clientHeight*d);ctx.clearRect(0,0,canvas.width,canvas.height);const s=canvas.width/view[2],p=screen([0,D.rows*D.cell]);ctx.imageSmoothingEnabled=false;ctx.drawImage(background,p[0],p[1],D.cols*D.cell*s,D.rows*D.cell*s);
 const path=(points,color,width)=>{if(points.length<2)return;ctx.strokeStyle=color;ctx.lineWidth=width*d;ctx.beginPath();points.forEach((p,i)=>ctx[i?'lineTo':'moveTo'](...screen(p)));ctx.stroke();};
 for(const l of D.lines)path(l.p,l.type===0?'#333':'#3766c6',.8);
 const selected=new Set();for(const id of chain){const n=nodeById.get(id);if(n){selected.add(n.crest);selected.add(n.toe);}}
 for(const id of selected){const l=lineById.get(id);if(l)path(l.p,'#c09500',2.5);}
 if($('showSections').checked)for(const s of relevantSections())path([s.a,s.b],'#9b3bb4',1.5);
 const r=current();for(const [id,a,b,color,label] of [[r.upper,r.upper_start,r.upper_end,'#ed6b00','上部坡顶'],[r.lower,r.lower_start,r.lower_end,'#00a2b8','下部坡底']]){const l=lineById.get(id),ps=clipped(l,a,b);path(ps,color,5);if(ps.length){for(const point of [ps[0],ps.at(-1)]){ctx.fillStyle=color;ctx.beginPath();ctx.arc(...screen(point),5*d,0,Math.PI*2);ctx.fill();}const mid=screen(interpolate(l,(a+b)/2));ctx.font=`bold ${14*d}px sans-serif`;ctx.fillStyle='white';ctx.fillRect(mid[0],mid[1]-18*d,155*d,22*d);ctx.fillStyle=color;ctx.fillText(`${label} ${id}`,mid[0]+3*d,mid[1]-2*d);}}
}
function focus(){const r=current(),points=[...clipped(lineById.get(r.upper),r.upper_start,r.upper_end),...clipped(lineById.get(r.lower),r.lower_start,r.lower_end)];if(!points.length)return;const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);view=[(xmin+xmax)/2,(ymin+ymax)/2,Math.max(20,xmax-xmin,(ymax-ymin)*canvas.clientWidth/Math.max(1,canvas.clientHeight))*1.2];draw();}
function boundaryChanged(role){const l=lineById.get(Number($(role).value));if(l){$(role==='upper'?'uStart':'lStart').value=0;$(role==='upper'?'uEnd':'lEnd').value=l.length;}paths=[];$('paths').replaceChildren();option($('paths'),'','边界已变更，请查找路径');invalidate();}
function findPaths(){paths=[];const r=current(),starts=D.nodes.filter(n=>n.crest===r.upper);let visits=0,truncated=false;
 function visit(id,path){if(++visits>20000||paths.length>=128){truncated=true;return;}if(path.includes(id))return;const seq=[...path,id];if(nodeById.get(id).toe===r.lower){if(seq.length>=2)paths.push(seq);return;}for(const e of D.edges)if(e[0]===id)visit(e[1],seq);}
 for(const n of starts)visit(n.id,[]);$('paths').replaceChildren();option($('paths'),'','请选择路径');paths.forEach((p,i)=>option($('paths'),i,p.map(id=>`${nodeById.get(id).crest}→${nodeById.get(id).toe}`).join(' / ')));
 if(paths.length===1){chain=[...paths[0]];$('paths').value=0;invalidate('找到唯一有支持的台阶链，仍需核实完整范围。');}
 else message(paths.length?`找到 ${paths.length} 条路径，请明确选择，不能自动认定。${truncated?'搜索已截断，列表不完整。':''}`:`未找到可靠链。可调整边界或手动编辑草稿；不强制跨断点连接。${truncated?'搜索达到上限。':''}`);
}
)HTML";
    out<<R"HTML(
function save(confirmed){const r=current(),error=validate(r,confirmed);if(error){message(error);return false;}if(records.has(r.id)&&r.id!==activeSaved){message('编号已存在，请换一个编号或选择该记录编辑。');return false;}
 if(activeSaved&&activeSaved!==r.id)records.delete(activeSaved);r.status=confirmed?'range_confirmed':'draft';r.confirmed_at=confirmed?new Date().toISOString():null;
 records.set(r.id,copy(r));activeSaved=r.id;state=r.status;confirmedAt=r.confirmed_at;refreshSaved();render();message(confirmed?'范围已由你确认并保存到页面内存。请下载 JSON 备份；此确认不等于参数或安全评价已完成。':'草稿已保存到页面内存，请下载 JSON。');return true;}
function download(name,text,type){const url=URL.createObjectURL(new Blob([text],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function payload(){return {schema:'workface-range-review/v1',dataset:D.dataset,scope:'manual_range_only_not_safety_or_geometry_approval',exported_at:new Date().toISOString(),records:[...records.values()]};}
function restore(p){if(!p||p.schema!=='workface-range-review/v1'||p.dataset!==D.dataset||!Array.isArray(p.records)||p.records.length>1000)throw Error('记录版本或数据指纹不匹配，拒绝导入。');const next=new Map();
 for(const raw of p.records){if(!raw||typeof raw!=='object')throw Error('无效记录');const r=copy(raw),error=validate(r,r.status==='range_confirmed');if(error)throw Error(error);if(!['draft','range_confirmed'].includes(r.status)||next.has(r.id))throw Error('状态无效或编号重复');if(typeof r.source_candidate!=='string'||(r.source_candidate&&!D.candidates.some(c=>c.name===r.source_candidate)))throw Error('未知来源候选');if(r.status==='range_confirmed'&&(typeof r.confirmed_at!=='string'||!Number.isFinite(Date.parse(r.confirmed_at))))throw Error('确认时间无效');next.set(r.id,r);}
 // Validate every row before touching existing records.
 for(const id of next.keys())if(records.has(id))throw Error('导入编号与当前记录重复，请先删除同名记录或在新打开的页面恢复。');
 for(const [id,r] of next)records.set(id,r);if(next.size){const r=next.values().next().value;activeSaved=r.id;setForm(r);}refreshSaved();message(`已恢复 ${next.size} 条人工记录；确认状态来自导入文件，不是程序自动认定。`);}
$('candidate').onchange=loadCandidate;$('upper').onchange=()=>boundaryChanged('upper');$('lower').onchange=()=>boundaryChanged('lower');
for(const id of ['name','reviewer','note','uStart','uEnd','lStart','lEnd'])$(id).oninput=()=>invalidate();
for(const id of ['checkTop','checkBottom','checkChain'])$(id).onchange=()=>{if(state==='range_confirmed')invalidate('确认项变化，已撤销确认。');};
$('find').onclick=findPaths;$('paths').onchange=()=>{if($('paths').value!==''){chain=[...paths[Number($('paths').value)]];invalidate();}};
$('append').onclick=()=>{const id=Number($('node').value);if(!nodeById.has(id)||chain.includes(id)){message('节点无效或已存在。');return;}chain.push(id);invalidate();};
$('saveDraft').onclick=()=>save(false);$('confirm').onclick=()=>save(true);$('revoke').onclick=()=>invalidate('确认已撤销，请重新核验。');
$('saved').onchange=()=>{const r=records.get($('saved').value);if(r){activeSaved=r.id;setForm(copy(r));message('已加载保存记录；修改范围会撤销确认。');focus();}};
$('delete').onclick=()=>{if(activeSaved&&records.has(activeSaved)){records.delete(activeSaved);activeSaved=null;invalidate('记录已从页面内存移除，已下载的文件不受影响。');refreshSaved();}};
$('new').onclick=()=>{activeSaved=null;let i=1;while(records.has('WF_'+i))i++;setForm({id:'WF_'+i,upper:-1,lower:-1,upper_start:0,upper_end:0,lower_start:0,lower_end:0,chain:[],source_candidate:'',reviewer:'',note:'',status:'draft',confirmed_at:null,checks:{}});refreshSaved();message('请选择上部坡顶、下部坡底，再查找台阶链。');};
$('export').onclick=()=>{if(!records.size){message('请先保存草稿或确认范围。');return;}download('workface_ranges.json',JSON.stringify(payload(),null,2),'application/json');message('已下载保存记录。尚未点击保存的编辑不会写入 JSON。');};
$('import').onchange=async()=>{const file=$('import').files[0];if(!file)return;try{if(file.size>5e6)throw Error('文件过大');restore(JSON.parse(await file.text()));}catch(e){message('导入失败：'+e.message);}finally{$('import').value='';}};
$('localSave').onclick=()=>{try{if(!records.size)throw Error('请先保存记录');localStorage.setItem('workface-range:'+D.dataset,JSON.stringify(payload()));message('已备份到本浏览器。仍建议下载 JSON 防止浏览器清理数据。');}catch(e){message('备份失败：'+e.message);}};
$('localLoad').onclick=()=>{try{const text=localStorage.getItem('workface-range:'+D.dataset);if(!text)throw Error('没有此数据集备份');restore(JSON.parse(text));}catch(e){message('恢复失败：'+e.message);}};
$('preview').onclick=()=>{const accepted=[...records.values()].filter(r=>r.status==='range_confirmed');if(!accepted.length){message('没有已确认范围。');return;}download('workfaces.csv','workface_id,upper_line_id,lower_line_id,start_m,end_m,spacing_m,max_distance_m\n'+accepted.map(r=>[r.id,r.upper,r.lower,r.upper_start,r.upper_end,5,1500].join(',')).join('\n')+'\n','text/csv');message('CSV 仅兼容旧版预览：不携带下边界区段、台阶链和确认状态；完整记录请使用 JSON。');};
$('fit').onclick=()=>{view=[D.cols*D.cell/2,D.rows*D.cell/2,Math.max(D.cols*D.cell,D.rows*D.cell*canvas.clientWidth/canvas.clientHeight)*1.04];draw();};$('focus').onclick=focus;$('showSections').onchange=draw;
function click(x,y){const p=world(x,y),role=$('pick').value;let found=null,best=12*view[2]/canvas.clientWidth;
 if(['uStart','uEnd','lStart','lEnd'].includes(role)){const l=lineById.get(Number($(role[0]==='u'?'upper':'lower').value));if(!l){message('请先选择对应边界');return;}const hit=station(l,p);if(hit.d>best){message('请点击已选边界附近。');return;}$(role).value=hit.s.toFixed(3);invalidate();return;}
 for(const l of D.lines){if((role==='upper'&&l.type!==0)||(role==='lower'&&l.type!==1))continue;const hit=station(l,p);if(hit.d<best){best=hit.d;found=l;}}
 if(!found){message('未点中边界，请放大后重试。');return;}if(role==='inspect'){message(`${found.type===0?'坡顶':'坡底'} ${found.id}，线长 ${found.length.toFixed(2)} m。`);return;}$(role).value=found.id;boundaryChanged(role);
}
canvas.onpointerdown=e=>{canvas.setPointerCapture(e.pointerId);drag={x:e.offsetX,y:e.offsetY,view:[...view],moved:false};};canvas.onpointermove=e=>{if(!drag)return;const dx=e.offsetX-drag.x,dy=e.offsetY-drag.y;if(Math.hypot(dx,dy)>4)drag.moved=true;if(drag.moved){view[0]=drag.view[0]-dx*view[2]/canvas.clientWidth;view[1]=drag.view[1]+dy*view[2]/canvas.clientWidth;draw();}};canvas.onpointerup=e=>{if(drag&&!drag.moved)click(e.offsetX,e.offsetY);drag=null;};canvas.onpointercancel=()=>drag=null;
canvas.onwheel=e=>{e.preventDefault();view[2]=Math.max(10,Math.min(D.cols*D.cell*3,view[2]*Math.exp(e.deltaY*.001)));draw();};
new ResizeObserver(draw).observe(canvas);render();
</script></html>)HTML";
    out.flush();if(!out)throw std::runtime_error("Cannot finish range review output");
}
}
