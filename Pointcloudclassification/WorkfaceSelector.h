#pragma once
#include "GroupedWorkfaceReview.h"
#include "ResolvedWorkfaceReview.h"
#include "ContinuousWorkfaces.h"
#include "OverallSlopePreview.h"
#include "PaperRiskRules.h"
#include "WorkfaceRangeReview.h"

namespace overall {
inline void writeSelector(const std::vector<FeatureLine>& lines,const std::vector<std::vector<GridCell>>& grid,
    double cell,const Eigen::Vector4f& origin,const std::vector<Preview>& previews,const std::string& prefix){
    if(grid.empty()||grid[0].empty())throw std::runtime_error("Cannot display empty grid");
    auto local=localLines(lines,origin);
    std::ofstream f(prefix+"workface_selector.html"),inventory(prefix+"workface_boundaries.csv");
    if(!f||!inventory)throw std::runtime_error("Cannot write workface selector");
    inventory<<std::fixed<<std::setprecision(8)<<"line_id,type,horizontal_length,min_z,max_z\n";
    f<<R"HTML(<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>多台阶边坡几何测量（待审核）</title>
<style>body{font:16px system-ui,sans-serif;margin:18px;color:#162338;background:#f4f7fa}h1{font-size:22px}label{display:inline-flex;gap:6px;align-items:center;margin:5px;flex-wrap:wrap}button,select,input{font:inherit;padding:6px}input{width:90px}button{cursor:pointer}canvas{display:block;width:100%;height:65vh;background:#fff;touch-action:none;border:1px solid #ccd5df}#status{min-height:26px;color:#8c2f00}#config{white-space:pre-wrap}p{max-width:1000px}#profile{height:220px}table{border-collapse:collapse}td,th{padding:6px;text-align:left;border-bottom:1px solid #ccd5df}</style>
<h1>多台阶边坡几何测量（待审核，不是正式工作帮坡角）</h1>
<p>自动候选由已有台阶配对及可靠平台连接生成。候选范围是观测关系图的范围，不保证完整工作帮，也不表示已确认采剥状态；未找到连接可能是漏配而非真实边界。AUTO 候选仍须核验。</p>
<label>查看候选/分组<select id="candidate"><option value="">全部</option></select></label><label><input id="showRejected" type="checkbox">显示自动候选中的异常截面</label><span id="candidateDetail"></span>
<p>黑：坡顶；蓝：坡底。上下端均可选坡顶或坡底，实际高程必须上高下低。点击地图选线，滚轮缩放、拖动平移。线编号只适用于本次特征提取结果。交叉及其他异常截面不输出倾角。</p>
<div><label>点击选择<select id="role"><option value="slope">坡面配对查看</option><option value="upper">上端边界</option><option value="lower">下端边界</option></select></label><label>地图选线类型<select id="pickType"><option value="0">坡顶</option><option value="1">坡底</option></select></label><label>上端<select id="upper"><option value="">请选择</option></select></label><label>下端<select id="lower"><option value="">请选择</option></select></label><button id="reset">全图</button></div>
<p>默认点击红色坡面查看已有单台阶配对：橙色为坡顶，青色为坡底，紫色为配对测线。此操作不会修改整体边坡上下端选择。只显示附近已有合格配对，不自动补配或按编号推断。</p><p id="slopeDetail" role="status" aria-live="polite">尚未选择坡面。</p>
<div><label>工作帮编号<input id="name" value="workface_1" style="width:140px"></label><label>起点/m<input id="start" type="number" value="0" min="0"></label><label>终点/m<input id="end" type="number" value="-1"><span>-1 为线末端</span></label><label>间距/m<input id="spacing" type="number" value="5" min="1"></label><label>搜索上限/m<input id="limit" type="number" value="1500" min="1" max="10000"></label></div>
<div><button id="add">加入预览配置</button><button id="download">下载 workfaces.csv</button><label><input id="showSections" type="checkbox" checked>显示本次生成的截面（绿色待确认，红色需检查）</label></div>
<p id="status" role="status" aria-live="polite">选择的边界不自动代表真实工作帮上下端。裁切边界和缺测区需要人工确认。</p>
<canvas id="map" aria-label="完整分类网格和可选择的坡顶坡底边界"></canvas>
<div id="config"></div><p>下载配置后，将其拖到 preview_workfaces.cmd（需已运行过点云处理），或将配置路径作为主程序第二个参数。重新运行后本页面将显示新的截面与剖面。不会自动生成正式工作帮坡角。</p>
<label>查看已生成截面<select id="section"><option value="">请选择</option></select></label><p id="detail"></p><canvas id="profile" aria-label="所选截面的观测高程剖面，缺测处断开"></canvas>
<script>
const DATA={"cell":)HTML"<<cell<<",\"rows\":"<<grid.size()<<",\"cols\":"<<grid[0].size()<<",\"lines\":[";
    bool first=true;
    for(auto& l:local){if(!first)f<<',';first=false;double low=l.p[0].z(),high=low;
        for(auto& p:l.p){low=std::min(low,p.z());high=std::max(high,p.z());}
        inventory<<l.id<<','<<l.type<<','<<l.s.back()<<','<<low<<','<<high<<'\n';
        f<<"{\"id\":"<<l.id<<",\"type\":"<<l.type<<",\"length\":"<<l.s.back()<<",\"points\":[";
        for(size_t i=0;i<l.p.size();++i){if(i)f<<',';f<<'['<<l.p[i].x()<<','<<l.p[i].y()<<']';}f<<"]}";
    }
    f<<"],\"runs\":[";first=true;
    for(size_t r=0;r<grid.size();++r){size_t c=0;while(c<grid[r].size()){
        int type=grid[r][c].count>0?grid[r][c].type:-1;size_t end=c+1;
        while(end<grid[r].size()&&(grid[r][end].count>0?grid[r][end].type:-1)==type)++end;
        if(!first)f<<',';first=false;f<<'['<<r<<','<<c<<','<<end-c<<','<<type<<']';c=end;
    }}
    f<<"],\"benchPairs\":[";first=true;
    // Use the same single-bench matching algorithm as the measurement pipeline.
    const auto benchPairs=sections::measure(lines,grid,cell,origin);
    for(const auto& s:benchPairs)if(s.status=="matched"){
        if(!first)f<<',';first=false;
        f<<"{\"id\":"<<s.id<<",\"crest\":"<<s.crestLine<<",\"toe\":"<<s.toeLine
         <<",\"a\":["<<s.crest.x()-origin.x()<<','<<s.crest.y()-origin.y()<<"],\"b\":["<<s.toe.x()-origin.x()<<','<<s.toe.y()-origin.y()<<"]}";
    }
    f<<"],\"sections\":[";first=true;
    for(auto& a:previews){if(!first)f<<',';first=false;
        f<<"{\"id\":"<<a.id<<",\"name\":\""<<a.name<<"\",\"status\":\""<<a.status<<"\",\"upper\":"<<a.upper<<",\"lower\":"<<a.lower
         <<",\"station\":"<<a.station<<",\"coverage\":"<<a.coverage<<",\"runs\":"<<a.slopeRuns<<",\"width\":"<<a.width<<",\"target\":"<<(a.hasTarget?"true":"false")
         <<",\"upperType\":"<<a.upperType<<",\"lowerType\":"<<a.lowerType<<",\"height\":";
        if(a.geometryAvailable)f<<a.height;else f<<"null";
        f<<",\"angle\":";if(a.geometryAvailable)f<<a.angle;else f<<"null";
        f<<",\"top\":["<<a.top.x()-origin.x()<<','<<a.top.y()-origin.y()<<"],\"bottom\":["<<a.bottom.x()-origin.x()<<','<<a.bottom.y()-origin.y()<<"],\"profile\":[";
        for(size_t i=0;i<a.profile.size();++i){if(i)f<<',';auto p=a.profile[i];f<<'['<<p.distance<<',';if(p.observed)f<<p.z;else f<<"null";f<<']';}f<<"]}";
    }
    f<<R"HTML(]};
const $=id=>document.getElementById(id), map=$('map'),ctx=map.getContext('2d'),pc=$('profile'),px=pc.getContext('2d');
const configs=[];let zoom=1,panX=0,panY=0,drag=null,transform,pickedBench=null;
const slopeGrid=new Uint8Array(DATA.rows*DATA.cols);
for(const [r,c,n,t] of DATA.runs)if(t===2)slopeGrid.fill(1,r*DATA.cols+c,r*DATA.cols+c+n);
function gridIndex(p){const c=Math.floor(p[0]/DATA.cell),r=Math.floor(p[1]/DATA.cell);return c>=0&&c<DATA.cols&&r>=0&&r<DATA.rows?r*DATA.cols+c:-1;}
function segmentDistance(p,a,b){const dx=b[0]-a[0],dy=b[1]-a[1],n=dx*dx+dy*dy,t=n?Math.max(0,Math.min(1,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/n)):0;return Math.hypot(p[0]-a[0]-t*dx,p[1]-a[1]-t*dy);}
function pickSlope(world){pickedBench=null;const seed=gridIndex(world);
 if(seed<0||!slopeGrid[seed]){$('slopeDetail').textContent='此处不是有观测的坡面，请点击红色坡面。';draw();return;}
 const mask=new Uint8Array(slopeGrid.length),queue=[seed];mask[seed]=1;
 for(let i=0;i<queue.length;i++){const v=queue[i],c=v%DATA.cols;for(const n of [c>0?v-1:-1,c+1<DATA.cols?v+1:-1,v-DATA.cols,v+DATA.cols])if(n>=0&&n<mask.length&&slopeGrid[n]&&!mask[n]){mask[n]=1;queue.push(n);}}
 const candidates=DATA.benchPairs.filter(s=>{const count=Math.max(1,Math.ceil(Math.hypot(s.b[0]-s.a[0],s.b[1]-s.a[1])/DATA.cell));for(let k=0;k<=count;k++){const t=k/count;if(mask[gridIndex([s.a[0]+t*(s.b[0]-s.a[0]),s.a[1]+t*(s.b[1]-s.a[1])])])return true;}return false;}).map(s=>({s,d:segmentDistance(world,s.a,s.b)})).sort((a,b)=>a.d-b.d);
 const best=candidates[0],limit=Math.max(5,3*DATA.cell);
 if(!best||best.d>limit){$('slopeDetail').textContent='此坡面点击位置附近没有合格的单台阶配对；不强行指定坡顶坡底。';draw();return;}
 if(candidates.some(v=>v.d<=best.d+DATA.cell&&(v.s.crest!==best.s.crest||v.s.toe!==best.s.toe))){$('slopeDetail').textContent='此处靠近多组配对边界，关系不唯一，请放大后点击坡面内部其他位置。';draw();return;}
 const pair=best.s,near=candidates.filter(v=>v.s.crest===pair.crest&&v.s.toe===pair.toe).map(v=>v.s);
 pickedBench={crest:pair.crest,toe:pair.toe,pairs:near};
 $('slopeDetail').textContent=`坡顶 ${pair.crest}（橙） ↔ 坡底 ${pair.toe}（青）；同一观测坡面内有 ${near.length} 条该组合的合格测线，最近测线 ${pair.id} 距点击位置 ${best.d.toFixed(2)} m。依据是单台阶测量配对，不是编号相邻；整条边界的其他地段不保证同属此坡面。`;
 draw();}
const palette={'-1':'#f1f1f1',0:'#ddd',1:'#a0cba3',2:'#e3a4a0',4:'#ffdb49'};
for(const l of DATA.lines)for(const role of ['upper','lower']){const o=document.createElement('option');o.value=l.id;o.textContent=l.id+' / '+(l.type===0?'坡顶':'坡底')+'（'+l.length.toFixed(1)+'m）';$(role).append(o);}
for(const s of DATA.sections){const o=document.createElement('option');o.value=s.id;o.textContent=s.name+' / '+s.id+' / '+s.status;$('section').append(o);}
const groupNames=[...new Set(DATA.sections.map(s=>s.name))],goodCount=name=>DATA.sections.filter(s=>s.name===name&&s.angle!==null).length;
groupNames.sort((a,b)=>goodCount(b)-goodCount(a));
for(const name of groupNames){const o=document.createElement('option');o.value=name;o.textContent=name+'（'+goodCount(name)+' 个可审核截面）';$('candidate').append(o);}
$('candidate').onchange=()=>{const name=$('candidate').value,rows=DATA.sections.filter(s=>!name||s.name===name);$('candidateDetail').textContent=` ${rows.length} 个采样截面，${rows.filter(s=>s.angle!==null).length} 个可审核几何值；范围及生产状态未确认。`;if(name&&rows.length){const first=rows.find(s=>s.angle!==null)||rows[0];$('upper').value=first.upper;$('lower').value=first.lower;$('section').value=first.id;}draw();chart();};
function size(canvas){const rect=canvas.getBoundingClientRect(),d=devicePixelRatio||1;canvas.width=Math.round(rect.width*d);canvas.height=Math.round(rect.height*d);return {w:rect.width,h:rect.height,d};}
function draw(){const {w,h,d}=size(map);ctx.setTransform(d,0,0,d,0,0);ctx.clearRect(0,0,w,h);
const scale=Math.min(w/(DATA.cols*DATA.cell),h/(DATA.rows*DATA.cell))*zoom;
const x0=(w-DATA.cols*DATA.cell*scale)/2+panX,y0=(h-DATA.rows*DATA.cell*scale)/2+panY;
transform={scale,x0,y0};ctx.save();ctx.translate(x0,y0);ctx.scale(scale,scale);
for(const [r,c,n,t] of DATA.runs){ctx.fillStyle=palette[t]||'#ddd';ctx.fillRect(c*DATA.cell,(DATA.rows-1-r)*DATA.cell,n*DATA.cell,DATA.cell);}
function path(points,color,width){ctx.strokeStyle=color;ctx.lineWidth=width/scale;ctx.beginPath();points.forEach((p,i)=>ctx[i?'lineTo':'moveTo'](p[0],DATA.rows*DATA.cell-p[1]));ctx.stroke();}
for(const l of DATA.lines){const selected=String(l.id)===$('upper').value||String(l.id)===$('lower').value;path(l.points,selected?'#d50072':l.type===0?'#111':'#1749d5',selected?4:1.1);}
if($('showSections').checked)for(const s of DATA.sections)if(s.target&&(!$('candidate').value||s.name===$('candidate').value)&&(!s.name.startsWith('AUTO_')||s.angle!==null||$('showRejected').checked))path([s.top,s.bottom],s.status==='boundary_confirmation_required'?'#00864a':'#b73719',String(s.id)===$('section').value?4:1);
if(pickedBench){for(const l of DATA.lines)if(l.id===pickedBench.crest||l.id===pickedBench.toe){path(l.points,l.id===pickedBench.crest?'#ff7900':'#00a8c6',4);const p=l.points[Math.floor(l.points.length/2)];ctx.save();ctx.translate(p[0],DATA.rows*DATA.cell-p[1]);ctx.scale(1/scale,1/scale);ctx.translate(0,l.type===0?-18:24);ctx.font='bold 15px sans-serif';ctx.fillStyle='#fff';ctx.fillRect(-3,-17,90,22);ctx.fillStyle='#111';ctx.fillText((l.type===0?'坡顶 ':'坡底 ')+l.id,0,0);ctx.restore();}for(const s of pickedBench.pairs)path([s.a,s.b],'#8a168f',2);}
ctx.restore();}
function chart(){const {w,h,d}=size(pc);px.setTransform(d,0,0,d,0,0);px.clearRect(0,0,w,h);const s=DATA.sections.find(s=>String(s.id)===$('section').value);if(!s){$('detail').textContent='请选择截面；所有截面均待人工确认。';return;}
$('detail').textContent=`截面 ${s.id}：${s.status}；上端 ${s.upperType===0?'坡顶':'坡底'} ${s.upper} → 下端 ${s.lowerType===0?'坡顶':'坡底'} ${s.lower}；水平跨度 ${s.target?s.width.toFixed(2):'未测得'} m；`+(s.angle===null?'未通过检查，不输出高差和倾角。':`高差 ${s.height.toFixed(2)} m；选定边界间整体倾角 ${s.angle.toFixed(2)}°（待审核）。`)+`观测覆盖 ${(s.coverage*100).toFixed(1)}%；坡面连续段 ${s.runs}（不是精确台阶数）。`;
const valid=s.profile.filter(p=>p[1]!==null);if(!valid.length)return;const zs=valid.map(p=>p[1]),lo=Math.min(...zs)-1,hi=Math.max(...zs)+1;
const x=t=>65+t/Math.max(1,s.width)*(w-90),y=z=>h-40-(z-lo)/(hi-lo)*(h-65);
px.font='13px sans-serif';px.fillStyle='#162338';px.strokeStyle='#bac6d2';for(let i=0;i<=4;i++){const z=lo+(hi-lo)*i/4;px.beginPath();px.moveTo(65,y(z));px.lineTo(w-20,y(z));px.stroke();px.fillText(z.toFixed(1),5,y(z)+4);px.fillText((s.width*i/4).toFixed(0),x(s.width*i/4)-8,h-22);}
px.fillText('高程/m',5,16);px.fillText('距上端水平距离/m',Math.max(65,w/2-65),h-4);px.strokeStyle='#165bc7';px.lineWidth=2;px.beginPath();let pen=false;
for(const p of s.profile){if(p[1]===null){pen=false;continue;}px[pen?'lineTo':'moveTo'](x(p[0]),y(p[1]));pen=true;}px.stroke();}
function selected(id){return DATA.lines.find(l=>String(l.id)===$(id).value);}
function showConfigs(){const box=$('config');box.replaceChildren();for(let i=0;i<configs.length;i++){const row=document.createElement('div');row.textContent=configs[i].join(',')+' ';const b=document.createElement('button');b.textContent='移除';b.onclick=()=>{configs.splice(i,1);showConfigs();};row.append(b);box.append(row);}}
$('add').onclick=()=>{const up=selected('upper'),down=selected('lower'),name=$('name').value.trim(),start=Number($('start').value),end=Number($('end').value),spacing=Number($('spacing').value),limit=Number($('limit').value);
if(!up||!down||up.id===down.id||!/^[A-Za-z0-9_-]{1,64}$/.test(name)||configs.some(c=>c[0]===name)||![start,end,spacing,limit].every(Number.isFinite)||start<0||(end<0&&end!==-1)||start>(end===-1?up.length:end)||end>up.length||spacing<1||limit<=0||limit>10000){$('status').textContent='请选择两条不同边界；编号须唯一且仅含英文字母、数字、_、-；检查区段和间距。';return;}
configs.push([name,up.id,down.id,start,end,spacing,limit]);showConfigs();$('status').textContent='配置已加入。上下边界仍需你核对，下载后重新生成截面。';};
$('download').onclick=()=>{if(!configs.length){$('status').textContent='请先加入至少一组边界配置。';return;}const header='workface_id,upper_line_id,lower_line_id,start_m,end_m,spacing_m,max_distance_m';const blob=new Blob([header+'\n'+configs.map(c=>c.join(',')).join('\n')+'\n'],{type:'text/csv;charset=utf-8'});const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='workfaces.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
function clickAt(x,y){const world=[(x-transform.x0)/transform.scale,DATA.rows*DATA.cell-(y-transform.y0)/transform.scale];let best=10/transform.scale,found=null;
if($('role').value==='slope'){pickSlope(world);return;}
for(const l of DATA.lines){if(l.type!==Number($('pickType').value))continue;for(let i=1;i<l.points.length;i++){const a=l.points[i-1],b=l.points[i],vx=b[0]-a[0],vy=b[1]-a[1],n=vx*vx+vy*vy;if(!n)continue;const t=Math.max(0,Math.min(1,((world[0]-a[0])*vx+(world[1]-a[1])*vy)/n));const dist=Math.hypot(world[0]-a[0]-t*vx,world[1]-a[1]-t*vy);if(dist<best){best=dist;found=l;}}}
if(found){$($('role').value).value=found.id;if($('role').value==='upper'){$('start').value=0;$('end').value=-1;}$('status').textContent='已选择 '+($('role').value==='upper'?'上端':'下端')+(found.type===0?'坡顶':'坡底')+' '+found.id+'。';draw();}}
map.onpointerdown=e=>{map.setPointerCapture(e.pointerId);drag={x:e.offsetX,y:e.offsetY,px:panX,py:panY,moved:false};};
map.onpointermove=e=>{if(!drag)return;const dx=e.offsetX-drag.x,dy=e.offsetY-drag.y;if(Math.hypot(dx,dy)>4)drag.moved=true;if(drag.moved){panX=drag.px+dx;panY=drag.py+dy;draw();}};
map.onpointerup=e=>{if(drag&&!drag.moved)clickAt(e.offsetX,e.offsetY);drag=null;};map.onpointercancel=()=>drag=null;
map.onwheel=e=>{e.preventDefault();zoom=Math.max(1,Math.min(12,zoom*Math.exp(-e.deltaY*.001)));draw();};
$('reset').onclick=()=>{zoom=1;panX=panY=0;draw();};for(const id of ['upper','lower','showSections','showRejected'])$(id).onchange=draw;
$('upper').addEventListener('change',()=>{$('start').value=0;$('end').value=-1;});$('section').onchange=()=>{draw();chart();};
new ResizeObserver(()=>{draw();chart();}).observe(map);draw();chart();
</script></html>)HTML";
}
inline void run(const std::vector<FeatureLine>& lines,const std::vector<std::vector<GridCell>>& grid,double cell,
    const Eigen::Vector4f& origin,const std::string& config,const std::string& prefix){
    std::vector<Preview> previews;
    const auto slopes=sections::measure(lines,grid,cell,origin);
    const auto widths=platforms::measure(slopes,lines,grid,cell,origin);
    const auto graph=autowalls::discover(slopes,widths);
    auto automatic=autowalls::measure(graph,lines,grid,cell,origin);
    autowalls::save(graph,automatic,prefix);
    range_review::write(lines,grid,cell,origin,graph,automatic,prefix);
    const auto groupedGraph=grouped::discover(slopes,widths,lines,grid,cell,origin);
    const auto groupedRows=grouped::measure(groupedGraph,lines,grid,cell,origin);
    grouped::save(groupedGraph,groupedRows,prefix);
    grouped_review::write(lines,grid,cell,origin,groupedGraph,groupedRows,prefix);
    const auto resolution=resolved::run(groupedGraph,groupedRows,widths,lines,grid,cell,origin);
    resolved::save(resolution,groupedRows,prefix);
    continuous::process(groupedGraph,groupedRows,widths,lines,grid,cell,origin,prefix);
    resolved_review::write(lines,grid,cell,origin,groupedGraph,groupedRows,resolution,prefix);
    paper_rules::save(slopes,widths,automatic,prefix);
    writeSelector(lines,grid,cell,origin,automatic,prefix+"auto_");
    if(!config.empty())previews=measure(lines,grid,cell,origin,readSelections(config));else previews=automatic;
    save(previews,prefix);writeSelector(lines,grid,cell,origin,previews,prefix);
}
}
