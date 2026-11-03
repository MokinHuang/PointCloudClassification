/* Auditable geometry engine. XY metres, Z metres; no gap filling or line editing. */
const Engine=(()=>{
const EPS=1e-7, dot=(p,v)=>p[0]*v[0]+p[1]*v[1], lerp=(a,b,t)=>a.map((v,i)=>v+(b[i]-v)*t);
function axes(deg){let a=deg*Math.PI/180;return {d:[Math.cos(a),Math.sin(a)],n:[-Math.sin(a),Math.cos(a)]};}
function segmentParts(data,refs){const out=[];for(const ref of refs){const l=data.lines.find(l=>l.id===ref.id);for(let i=1;i<l.p.length;i++){let lo=Math.max(ref.start,l.s[i-1]),hi=Math.min(ref.end,l.s[i]);if(hi-lo<=EPS||l.s[i]-l.s[i-1]<=EPS)continue;out.push({a:lerp(l.p[i-1],l.p[i],(lo-l.s[i-1])/(l.s[i]-l.s[i-1])),b:lerp(l.p[i-1],l.p[i],(hi-l.s[i-1])/(l.s[i]-l.s[i-1])),id:l.id,segment:i-1,lo,hi,sa:l.s[i-1],sb:l.s[i]});}}return out;}
function validate(data,f){
 if(f.kind!==undefined&&! ["platform","workface"].includes(f.kind))throw Error("记录类型无效");
 if(f.kind==="platform"&&! ["unknown","working","temporary_limit","final_limit"].includes(f.purpose))throw Error("平台用途无效");
 if(!f||typeof f.name!=='string'||!f.name.trim()||f.name.length>100)throw Error('工作帮名称无效');
 if(!['angle','left','right','spacing'].every(k=>Number.isFinite(f[k]))||f.spacing<1||f.right<f.left||(f.right-f.left)/f.spacing>2000)throw Error('范围/间距无效（间距至少1m，最多2001剖面）');
 if(!Array.isArray(f.excluded)||f.excluded.some(x=>!Number.isInteger(x)||x<0||x>2000)||typeof f.confirmed!=='boolean'||(f.confirmed&&(typeof f.reviewer!=='string'||!f.reviewer.trim())))throw Error('剖面排除/确认状态无效');
 for(const [role,type] of [['upper',0],['lower',1]]){
 if(!Array.isArray(f[role])||f[role].length>1000)throw Error('边界列表无效');
 for(const r of f[role]){const l=data.lines.find(l=>l.id===r.id);if(!l||l.type!==type||r.hash!==l.hash||!Number.isFinite(r.start)||!Number.isFinite(r.end)||r.start<0||r.end>l.s.at(-1)+EPS||r.end<=r.start)throw Error('原始线段引用、类型或区段无效');}
 }
}
function intersections(parts,station,n){let hits=[],overlap=false;for(const q of parts){let a=dot(q.a,n)-station,b=dot(q.b,n)-station;if(Math.abs(a)<EPS&&Math.abs(b)<EPS){overlap=true;continue;}if(Math.abs(a-b)<EPS)continue;let t=a/(a-b);if(t<-EPS||t>1+EPS)continue;t=Math.max(0,Math.min(1,t));let p=lerp(q.a,q.b,t),s=q.lo+t*(q.hi-q.lo),ref={id:q.id,segment:q.segment,station:s,fraction:(s-q.sa)/(q.sb-q.sa)};
 // Adjacent segments of the SAME source line share one vertex. Distinct source lines remain ambiguous.
 const prior=hits.find(h=>h.ref.id===q.id&&Math.hypot(...h.p.map((v,i)=>v-p[i]))<EPS);if(!prior)hits.push({p,ref});}return {hits,overlap};}
function crosses(a,b,c,d){let v=[b[0]-a[0],b[1]-a[1]],w=[d[0]-c[0],d[1]-c[1]],z=[c[0]-a[0],c[1]-a[1]],cross=(x,y)=>x[0]*y[1]-x[1]*y[0],den=cross(v,w);if(Math.abs(den)<EPS){if(Math.abs(cross(z,v))>EPS)return false;let vv=dot(v,v);if(vv<EPS)return false;let t0=dot(z,v)/vv,t1=t0+dot(w,v)/vv;return Math.max(Math.min(t0,t1),0)<=Math.min(Math.max(t0,t1),1)+EPS;}let t=cross(z,w)/den,u=cross(z,v)/den;return t>=-EPS&&t<=1+EPS&&u>=-EPS&&u<=1+EPS;}
function makeGrid(data){return new Map(data.grid.map(g=>[g[0]*data.cols+g[1],g[2]]));}
function elevation(data,grid,p){let c=Math.floor(p[0]/data.cell),r=Math.floor(p[1]/data.cell);if(c<0||r<0||c>=data.cols||r>=data.rows)return null;return grid.get(r*data.cols+c)??null;}
function terrain(data,grid,a,b){let width=Math.hypot(b[0]-a[0],b[1]-a[1]),ts=[0,1];
 // Sample every crossed grid cell, including short corner slivers, not merely fixed-distance points.
 for(let dim=0;dim<2;dim++){let delta=b[dim]-a[dim];if(Math.abs(delta)<EPS)continue;let lo=Math.floor(Math.min(a[dim],b[dim])/data.cell)+1,hi=Math.floor(Math.max(a[dim],b[dim])/data.cell);for(let k=lo;k<=hi;k++){let t=(k*data.cell-a[dim])/delta;if(t>EPS&&t<1-EPS)ts.push(t);}}
 ts=[...new Set(ts)].sort((a,b)=>a-b);let profile=[],missing=0;for(let i=1;i<ts.length;i++){let t=(ts[i-1]+ts[i])/2,p=lerp(a,b,t),z=elevation(data,grid,p);profile.push([t*width,z]);if(z===null)missing+=(ts[i]-ts[i-1])*width;}
 profile.unshift([0,elevation(data,grid,a)]);profile.push([width,elevation(data,grid,b)]);return {profile,missingFraction:width?missing/width:1};}
function calculate(data,f,grid=makeGrid(data)){
 validate(data,f);let {d,n}=axes(f.angle),upper=segmentParts(data,f.upper),lower=segmentParts(data,f.lower),crossing=false;
 // Conservative rejection of a selected crossing; the user must crop out that geometry.
 for(const a of upper)for(const b of lower)if(crosses(a.a,a.b,b.a,b.b))crossing=true;
 let result=[];for(let i=0;i<=Math.floor((f.right-f.left)/f.spacing+EPS);i++){let station=f.left+i*f.spacing,u=intersections(upper,station,n),l=intersections(lower,station,n),r={index:i,station,status:'valid',upperHits:u.hits,lowerHits:l.hits,height:null,width:null,angle:null,profile:[],missingFraction:null,excluded:f.excluded.includes(i)};
 if(crossing)r.status='crossing_boundaries';else if(u.overlap||l.overlap)r.status='collinear_overlap';else if(!u.hits.length||!l.hits.length)r.status='missing_intersection';else if(u.hits.length!==1||l.hits.length!==1)r.status='multiple_intersections';
 if(r.status==='valid'){r.top=u.hits[0];r.bottom=l.hits[0];let a=r.top.p,b=r.bottom.p,w=dot(b,d)-dot(a,d),h=a[2]-b[2];let t=terrain(data,grid,a,b);r.profile=t.profile;r.missingFraction=t.missingFraction;if(w<=EPS||(f.kind!=='platform'&&h<=EPS))r.status='invalid_pair_order';else if(t.profile.some(p=>p[1]===null))r.status='terrain_missing';else {r.height=h;r.width=w;r.angle=Math.atan2(h,w)*180/Math.PI;if(f.kind==='platform'){let types=new Map(data.grid.map(g=>[g[0]*data.cols+g[1],g[3]])),num=Math.max(1,Math.ceil(w/data.cell)),supported=0;for(let k=0;k<num;k++){let p=lerp(a,b,(k+.5)/num),v=types.get(Math.floor(p[1]/data.cell)*data.cols+Math.floor(p[0]/data.cell));if(v===1||v===4)supported++;}r.platformSupport=supported/num;if(r.platformSupport<.8){r.status='platform_surface_unconfirmed';r.height=r.width=r.angle=null;}}}}
 result.push(r);}return result;
}
function summary(rows){let valid=rows.filter(r=>r.status==='valid'&&!r.excluded),missing=rows.filter(r=>r.status==='missing_intersection'||r.status==='terrain_missing').length,stats={};for(let key of ['height','width','angle']){let v=valid.map(r=>r[key]).sort((a,b)=>a-b);stats[key]=v.length?{min:v[0],max:v.at(-1),mean:v.reduce((a,b)=>a+b,0)/v.length,median:(v[Math.floor((v.length-1)/2)]+v[Math.floor(v.length/2)])/2}:null;}return {total:rows.length,valid:valid.length,excluded:rows.filter(r=>r.excluded).length,missing,missingRatio:rows.length?missing/rows.length:null,invalidRatio:rows.length?(rows.length-valid.length)/rows.length:null,stats};}
function restore(data,record){if(record.schema!=='manual-workface-v1'||record.dataset!==data.id)throw Error('数据集不匹配或记录版本不支持，未加载');if(!Array.isArray(record.faces)||record.faces.length>100)throw Error('工作帮记录无效');const faces=JSON.parse(JSON.stringify(record.faces));for(const f of faces)validate(data,f);if(new Set(faces.map(f=>f.id)).size!==faces.length||faces.some(f=>typeof f.id!=='string'))throw Error('工作帮ID无效');return faces;}
function sourceMeasurement(data,m){
 const r=m.row,upper=data.lines.find(l=>l.id===Number(r.crest_line_id)&&l.type===0),lower=data.lines.find(l=>l.id===Number(r.toe_line_id)&&l.type===1);if(!upper||!lower)return null;
 const s=Number(r.crest_station),i=upper.s.findIndex((v,i)=>i>0&&v>=s-EPS),j=Number(r.toe_segment),t=Number(r.toe_fraction);
 if(i<1||s<0||!Number.isInteger(j)||j<0||j+1>=lower.p.length||!Number.isFinite(t)||t<0||t>1)return null;
 const a=lerp(upper.p[i-1],upper.p[i],(s-upper.s[i-1])/(upper.s[i]-upper.s[i-1])),b=lerp(lower.p[j],lower.p[j+1],t),originalA=['crest_x','crest_y','crest_z'].map(k=>Number(r[k])),originalB=['toe_x','toe_y','toe_z'].map(k=>Number(r[k]));
 originalA[0]-=data.origin[0];originalA[1]-=data.origin[1];originalB[0]-=data.origin[0];originalB[1]-=data.origin[1];
 if([...a,...b,...originalA,...originalB].some(v=>!Number.isFinite(v))||Math.hypot(...a.map((v,k)=>v-originalA[k]))>.02||Math.hypot(...b.map((v,k)=>v-originalB[k]))>.02)return null;
 let width=Math.hypot(originalA[0]-originalB[0],originalA[1]-originalB[1]),height=originalA[2]-originalB[2],angle=Math.atan2(height,width)*180/Math.PI;
 if(m.kind==='bench'&&(!(height>0&&width>0)||Math.abs(width-Number(r.h_dist))>.001||Math.abs(height-Number(r.v_drop))>.001||Math.abs(angle-Number(r.angle))>.001))return null;
 if(m.kind==='platform'&&Math.abs(width-Number(r.platform_width))>.001)return null;
 return {a:originalA,b:originalB,height:m.kind==='bench'?Number(r.v_drop):null,angle:m.kind==='bench'?Number(r.angle):null,width:m.kind==='platform'?Number(r.platform_width):null,crestLine:upper.id,toeLine:lower.id,crestHash:upper.hash,toeHash:lower.hash};
}
function contained(a,b,f,rows){const {d,n}=axes(f.angle),sa=dot(a,n),sb=dot(b,n),ts=[0,1];if(Math.abs(sb-sa)>EPS)for(const r of rows){let t=(r.station-sa)/(sb-sa);if(t>0&&t<1)ts.push(t);}ts.sort((a,b)=>a-b);let checks=[...ts];for(let i=1;i<ts.length;i++)checks.push((ts[i-1]+ts[i])/2);
 return checks.every(t=>{let p=lerp(a,b,t),s=dot(p,n),v=dot(p,d);return rows.some((r,i)=>{let q=rows[i+1];if(!q||r.status!=='valid'||q.status!=='valid'||r.excluded||q.excluded||s<r.station-EPS||s>q.station+EPS)return false;let x=(s-r.station)/(q.station-r.station),lo=dot(r.top.p,d)*(1-x)+dot(q.top.p,d)*x,hi=dot(r.bottom.p,d)*(1-x)+dot(q.bottom.p,d)*x;return v>=lo-EPS&&v<=hi+EPS;});});
}
function associate(data,faces,calculations,grid=makeGrid(data)){
 return (data.measurements||[]).map(m=>{let checked=sourceMeasurement(data,m),out={source:m.source,sourceHash:m.sourceHash,sourceId:m.row.pair_id??m.row.sample_id,kind:m.kind,status:'source_geometry_unverified',faceId:null,height:null,angle:null,width:null};if(!checked)return out;
 out.evidence=checked;out.status='range_unresolved';if(terrain(data,grid,checked.a,checked.b).profile.some(p=>p[1]===null)){out.status='measurement_terrain_missing';return out;}
 let candidates=faces.filter((f,i)=>f.kind!=='platform'&&contained(checked.a,checked.b,f,calculations[i].rows));out.candidateFaces=candidates.map(f=>f.id);if(candidates.length!==1){out.status=candidates.length?'ambiguous_ownership':'outside_valid_corridor';return out;}if(!candidates[0].confirmed){out.status='range_not_confirmed';return out;}
 return {...out,status:'associated_geometry_only',faceId:candidates[0].id,height:checked.height,angle:checked.angle,width:checked.width};});
}
return {axes,dot,segmentParts,validate,calculate,summary,restore,makeGrid,elevation,intersections,crosses,associate,sourceMeasurement,contained};
})();
if(typeof module!=='undefined')module.exports=Engine;
