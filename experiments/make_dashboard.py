"""Build the self-contained TandemSpec results dashboard."""
import json, os, sys, statistics as st

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def load(name, default=None):
    p = f"{RES}/{name}"
    return json.load(open(p)) if os.path.exists(p) else default


def fmt(x, n=3):
    return "-" if x is None else f"{x:.{n}f}"


def build_data():
    s0 = load("stage0.json", {})
    meta = load("task_adapters.json", {})
    e1 = load("e1_acceptance_collapse.json", {"rows": []})
    e2 = load("e2_companion_repair.json", {"rows": []})
    e3 = load("e3_serving_model.json", {})
    e4 = load("e4_quant_mismatch.json", {"rows": []})

    kind = {int(k): v.get("kind", "in-dist") for k, v in meta.items()}
    for r in e1["rows"]:
        r["kind"] = kind.get(r["tenant"], "in-dist")

    beta0 = s0.get("base_acceptance", {}).get("beta_analytic")
    tok0 = s0.get("base_acceptance", {}).get("tokens_per_step_iid")

    # E2 aggregates: mean over tenants per arm
    arms, order = {}, []
    for r in e2["rows"]:
        arms.setdefault(r["arm"], []).append(r)
        if r["arm"] not in order:
            order.append(r["arm"])
    e2agg = [{
        "arm": a,
        "beta": st.mean(x["beta_analytic"] for x in arms[a]),
        "greedy": st.mean(x["beta_greedy"] for x in arms[a]),
        "tok": st.mean(x["tokens_per_step_iid"] for x in arms[a]),
        "params": arms[a][0]["extra_params"],
        "bytes": arms[a][0]["extra_bytes_fp16"],
        "train_s": st.mean(x["train_s"] for x in arms[a]),
    } for a in order]

    e4agg = {}
    for r in e4["rows"]:
        e4agg.setdefault(r["arm"], []).append(r)
    e4rows = [{
        "arm": a,
        "drift": st.mean(x["drift_from_reference_tvd"] for x in v),
        "shared": st.mean(x["beta_shared"] for x in v),
        "companion": st.mean(x["beta_companion"] for x in v),
        "tok_shared": st.mean(x["tok_step_shared"] for x in v),
        "tok_comp": st.mean(x["tok_step_companion"] for x in v),
    } for a, v in e4agg.items()]

    return {"stage0": s0, "beta0": beta0, "tok0": tok0, "meta": meta,
            "e1": e1["rows"], "e2": e2agg, "e2raw": e2["rows"],
            "e3": e3, "e4": e4rows, "config": e1.get("config", s0.get("config", {}))}


HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TandemSpec — companion draft adapters for multi-tenant speculative decoding</title>
<style>
:root{color-scheme:light dark}
.viz-root{
  --surface-0:#f6f6f4; --surface-1:#fcfcfb; --surface-2:#efeeea;
  --border:#dedcd6; --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#79776f;
  --series-1:#2a78d6; --series-2:#eb6834; --series-3:#1baf7a; --series-4:#eda100;
  --neutral:#9a988f; --good:#1baf7a; --bad:#e34948;
  --grid:#e6e4de;
}
@media (prefers-color-scheme: dark){
 :root:where(:not([data-theme="light"])) .viz-root{
  --surface-0:#111110; --surface-1:#1a1a19; --surface-2:#232321;
  --border:#34342f; --text-primary:#fff; --text-secondary:#c3c2b7; --text-muted:#8f8e84;
  --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70; --series-4:#c98500;
  --neutral:#6f6e66; --grid:#2c2c29;
 }}
:root[data-theme="dark"] .viz-root{
  --surface-0:#111110; --surface-1:#1a1a19; --surface-2:#232321;
  --border:#34342f; --text-primary:#fff; --text-secondary:#c3c2b7; --text-muted:#8f8e84;
  --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70; --series-4:#c98500;
  --neutral:#6f6e66; --grid:#2c2c29;
}
*{box-sizing:border-box}
body{margin:0;font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
/* surfaces and ink live on .viz-root, where the custom properties are defined --
   putting them on body would resolve against an undefined variable */
.viz-root{background:var(--surface-0);color:var(--text-primary);min-height:100vh;display:block}
.wrap{max-width:1080px;margin:0 auto;padding:32px 20px 80px}
header{border-bottom:1px solid var(--border);padding-bottom:20px;margin-bottom:28px}
h1{font-size:26px;margin:0 0 6px;letter-spacing:-.01em}
h2{font-size:19px;margin:40px 0 4px;letter-spacing:-.01em}
h3{font-size:14px;margin:24px 0 8px;color:var(--text-secondary);font-weight:600}
p.sub{color:var(--text-secondary);margin:0;max-width:70ch}
p.note{color:var(--text-secondary);font-size:13.5px;max-width:78ch;margin:8px 0 18px}
.hero{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:22px 0 8px}
.tile{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.tile .k{font-size:11.5px;text-transform:uppercase;letter-spacing:.06em;color:var(--text-muted)}
.tile .v{font-size:26px;font-weight:650;letter-spacing:-.02em;margin-top:4px}
.tile .d{font-size:12.5px;color:var(--text-secondary);margin-top:3px}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;padding:18px 18px 12px;margin:14px 0}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin:2px 0 10px;font-size:12.5px;color:var(--text-secondary)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.swatch{width:10px;height:10px;border-radius:3px;display:inline-block}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}
.grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:6px}
th,td{text-align:right;padding:6px 8px;border-bottom:1px solid var(--border)}
th:first-child,td:first-child{text-align:left}
th{color:var(--text-muted);font-weight:600;font-size:11.5px;text-transform:uppercase;letter-spacing:.05em}
tbody tr:hover{background:var(--surface-2)}
.toggle{background:none;border:1px solid var(--border);color:var(--text-secondary);
 border-radius:6px;padding:3px 9px;font-size:12px;cursor:pointer;margin-bottom:6px}
.toggle:hover{background:var(--surface-2);color:var(--text-primary)}
.hidden{display:none}
svg{display:block;width:100%;overflow:visible}
.tip{position:fixed;pointer-events:none;background:var(--surface-1);border:1px solid var(--border);
 border-radius:8px;padding:7px 10px;font-size:12.5px;box-shadow:0 6px 20px rgba(0,0,0,.14);
 opacity:0;transition:opacity .1s;z-index:50;white-space:nowrap;color:var(--text-primary)}
.tip b{font-weight:650}
code{background:var(--surface-2);padding:1px 5px;border-radius:4px;font-size:12.5px}
.foot{margin-top:44px;padding-top:16px;border-top:1px solid var(--border);
 color:var(--text-muted);font-size:12.5px;max-width:78ch}
.themebtn{float:right;margin-top:-4px}
</style></head>
<body><div class="viz-root"><div class="wrap">
<div class="tip" id="tip"></div>
<header>
<button class="toggle themebtn" onclick="tgTheme()">theme</button>
<h1>TandemSpec</h1>
<p class="sub">A shared speculative drafter imitates the <b>base</b> model, but every request in a
multi-LoRA server is verified against <b>base&nbsp;+&nbsp;tenant&nbsp;adapter</b>. This measures what
that costs, and repairs it with a rank-4 companion adapter on the drafter.</p>
</header>
<div id="app"></div>
<div class="foot" id="foot"></div>
</div></div>
<script>
const D = __DATA__;
const tip = document.getElementById('tip');
const cvar = n => getComputedStyle(document.querySelector('.viz-root')).getPropertyValue(n).trim();
function tgTheme(){const r=document.documentElement;
  const cur=r.getAttribute('data-theme')|| (matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');
  r.setAttribute('data-theme',cur==='dark'?'light':'dark'); render();}
function showTip(e,html){tip.innerHTML=html;tip.style.opacity=1;
  const p=12;let x=e.clientX+p,y=e.clientY+p;
  if(x+tip.offsetWidth>innerWidth-8)x=e.clientX-tip.offsetWidth-p;
  if(y+tip.offsetHeight>innerHeight-8)y=e.clientY-tip.offsetHeight-p;
  tip.style.left=x+'px';tip.style.top=y+'px';}
function hideTip(){tip.style.opacity=0}
const N=(v,d=3)=>v==null?'—':(+v).toFixed(d);
const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

/* ---------- line chart ---------- */
function lineChart(o){
  const W=o.width||440,H=o.height||210,m={t:14,r:16,b:34,l:46};
  const iw=W-m.l-m.r, ih=H-m.t-m.b;
  const pts=o.series.flatMap(s=>s.points);
  const xs=pts.map(p=>p.x), ys=pts.map(p=>p.y);
  const x0=o.xDomain?o.xDomain[0]:Math.min(...xs), x1=o.xDomain?o.xDomain[1]:Math.max(...xs);
  let y0=o.yDomain?o.yDomain[0]:Math.min(...ys), y1=o.yDomain?o.yDomain[1]:Math.max(...ys);
  if(y0===y1){y0-=.05;y1+=.05}
  const X=v=>m.l+(v-x0)/((x1-x0)||1)*iw, Y=v=>m.t+ih-(v-y0)/((y1-y0)||1)*ih;
  const nY=o.yTicks||4, nX=o.xTicks||5;
  let g='';
  for(let i=0;i<=nY;i++){const v=y0+(y1-y0)*i/nY, y=Y(v);
    g+=`<line x1="${m.l}" y1="${y}" x2="${W-m.r}" y2="${y}" stroke="${cvar('--grid')}" stroke-width="1"/>`+
       `<text x="${m.l-8}" y="${y+4}" text-anchor="end" font-size="10.5" fill="${cvar('--text-muted')}">${o.yFmt?o.yFmt(v):N(v,2)}</text>`;}
  for(let i=0;i<=nX;i++){const v=x0+(x1-x0)*i/nX;
    g+=`<text x="${X(v)}" y="${H-m.b+16}" text-anchor="middle" font-size="10.5" fill="${cvar('--text-muted')}">${o.xFmt?o.xFmt(v):N(v,2)}</text>`;}
  o.series.forEach(s=>{
    const c=cvar(s.color);
    const d=s.points.map((p,i)=>(i?'L':'M')+X(p.x)+' '+Y(p.y)).join(' ');
    g+=`<path d="${d}" fill="none" stroke="${c}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" ${s.dash?'stroke-dasharray="5 4"':''}/>`;
    s.points.forEach(p=>{g+=`<circle cx="${X(p.x)}" cy="${Y(p.y)}" r="4" fill="${c}" stroke="${cvar('--surface-1')}" stroke-width="2"
      class="mk" data-tip="${esc(`<b>${s.name}</b><br>${o.xLabel||'x'} ${N(p.x,2)}<br>${o.yLabel||'y'} <b>${N(p.y,4)}</b>${p.note?'<br>'+p.note:''}`)}"/>`;});
    if(o.directLabel!==false&&s.points.length){const lp=s.points[s.points.length-1];
      g+=`<text x="${X(lp.x)+7}" y="${Y(lp.y)+4}" font-size="11" fill="${cvar('--text-secondary')}">${esc(s.name)}</text>`;}
  });
  if(o.rule!=null){const y=Y(o.rule);
    g+=`<line x1="${m.l}" y1="${y}" x2="${W-m.r}" y2="${y}" stroke="${cvar('--neutral')}" stroke-width="1.5" stroke-dasharray="4 4"/>`+
       `<text x="${m.l+4}" y="${y-5}" font-size="10.5" fill="${cvar('--text-muted')}">${esc(o.ruleLabel||'')}</text>`;}
  return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">${g}</svg>`;
}

/* ---------- horizontal bars ---------- */
function barChart(o){
  const rowH=o.rowH||26, W=o.width||520, labelW=o.labelW||168;
  const H=o.bars.length*rowH+18;
  const max=o.max||Math.max(...o.bars.map(b=>b.value))*1.02;
  const min=o.min!=null?o.min:0, iw=W-labelW-58;
  let g='';
  o.bars.forEach((b,i)=>{
    const y=i*rowH+6, w=Math.max(2,(b.value-min)/((max-min)||1)*iw), c=cvar(b.color||'--series-1');
    g+=`<text x="0" y="${y+13}" font-size="11.5" fill="${cvar('--text-secondary')}">${esc(b.label)}</text>`;
    g+=`<rect x="${labelW}" y="${y+2}" width="${w}" height="${rowH-10}" rx="4" fill="${c}"
        class="mk" data-tip="${esc(`<b>${b.label}</b><br>${o.valueLabel||'value'} <b>${N(b.value,4)}</b>${b.note?'<br>'+b.note:''}`)}"/>`;
    g+=`<text x="${labelW+w+7}" y="${y+13}" font-size="11.5" fill="${cvar('--text-primary')}">${o.fmt?o.fmt(b.value):N(b.value,4)}</text>`;
  });
  if(o.rule!=null){const x=labelW+(o.rule-min)/((max-min)||1)*iw;
    g+=`<line x1="${x}" y1="0" x2="${x}" y2="${H-14}" stroke="${cvar('--neutral')}" stroke-width="1.5" stroke-dasharray="4 4"/>`;
    g+=`<text x="${x+4}" y="${H-4}" font-size="10.5" fill="${cvar('--text-muted')}">${esc(o.ruleLabel||'')}</text>`;}
  return `<svg viewBox="0 0 ${W} ${H+6}" preserveAspectRatio="xMidYMid meet">${g}</svg>`;
}

/* ---------- scatter ---------- */
function scatter(o){
  const W=o.width||900,H=o.height||400,m={t:18,r:30,b:44,l:62};
  const iw=W-m.l-m.r, ih=H-m.t-m.b;
  const pts=o.series.flatMap(s=>s.points);
  const x0=0,x1=Math.max(...pts.map(p=>p.x))*1.05;
  const ys=pts.map(p=>p.y), pad=(Math.max(...ys)-Math.min(...ys))*0.08||0.02;
  const y0=o.yDomain?o.yDomain[0]:Math.min(...ys)-pad, y1=o.yDomain?o.yDomain[1]:Math.min(1,Math.max(...ys)+pad);
  const X=v=>m.l+(v-x0)/((x1-x0)||1)*iw, Y=v=>m.t+ih-(v-y0)/((y1-y0)||1)*ih;
  let g='';
  for(let i=0;i<=4;i++){const v=y0+(y1-y0)*i/4,y=Y(v);
    g+=`<line x1="${m.l}" y1="${y}" x2="${W-m.r}" y2="${y}" stroke="${cvar('--grid')}" stroke-width="1"/>`+
       `<text x="${m.l-8}" y="${y+4}" text-anchor="end" font-size="10.5" fill="${cvar('--text-muted')}">${N(v,2)}</text>`;}
  for(let i=0;i<=4;i++){const v=x0+(x1-x0)*i/4;
    g+=`<text x="${X(v)}" y="${H-m.b+16}" text-anchor="middle" font-size="10.5" fill="${cvar('--text-muted')}">${N(v,2)}</text>`;}
  if(o.theory){const a=o.theory;
    const ay0=Math.max(y0,Math.min(y1,a.y0)), ay1=Math.max(y0,Math.min(y1,a.y1));
    g+=`<path d="M${X(a.x0)} ${Y(ay0)} L${X(a.x1)} ${Y(ay1)}" stroke="${cvar('--neutral')}"
        stroke-width="1.5" stroke-dasharray="5 4" fill="none"/>`+
       `<text x="${X((a.x0+a.x1)*0.42)}" y="${Y((ay0+ay1)*0.42)-11}" text-anchor="middle" font-size="11" fill="${cvar('--text-muted')}">${esc(a.label)}</text>`;}
  o.series.forEach(s=>{const c=cvar(s.color);
    s.points.forEach(p=>{g+=`<circle cx="${X(p.x)}" cy="${Y(p.y)}" r="5" fill="${c}" stroke="${cvar('--surface-1')}" stroke-width="2"
      class="mk" data-tip="${esc(`<b>${s.name}</b><br>${o.xLabel} ${N(p.x,4)}<br>${o.yLabel} <b>${N(p.y,4)}</b>${p.note?'<br>'+p.note:''}`)}"/>`;});});
  g+=`<text x="${m.l+iw/2}" y="${H-2}" text-anchor="middle" font-size="11" fill="${cvar('--text-muted')}">${esc(o.xLabel)}</text>`;
  return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">${g}</svg>`;
}

function legend(items){return `<div class="legend">`+items.map(i=>
  `<span><i class="swatch" style="background:${cvar(i.color)}"></i>${esc(i.name)}</span>`).join('')+`</div>`;}
function table(head,rows,id){
  return `<button class="toggle" onclick="document.getElementById('${id}').classList.toggle('hidden')">table view</button>`+
   `<div id="${id}" class="hidden"><table><thead><tr>`+head.map(h=>`<th>${esc(h)}</th>`).join('')+
   `</tr></thead><tbody>`+rows.map(r=>`<tr>`+r.map(c=>`<td>${c}</td>`).join('')+`</tr>`).join('')+
   `</tbody></table></div>`;}

function render(){
  const A=[];
  const b0=D.beta0, t0=D.tok0;
  const s1=D.e1.filter(r=>r.strength===1.0);
  const mean=a=>a.length?a.reduce((x,r)=>x+r.beta_analytic,0)/a.length:null;
  const bIn=mean(s1.filter(r=>r.kind==='in-dist')), bOut=mean(s1.filter(r=>r.kind==='held-out'));
  const tok=a=>a.length?a.reduce((x,r)=>x+r.tokens_per_step_iid,0)/a.length:null;
  const tIn=tok(s1.filter(r=>r.kind==='in-dist')), tOut=tok(s1.filter(r=>r.kind==='held-out'));
  const bShared=mean(s1);
  const comp=D.e2.find(a=>a.arm==='companion-tvd-r4');
  const shared2=D.e2.find(a=>a.arm==='shared-drafter');
  const full=D.e2.find(a=>a.arm==='full-ft-tvd');

  /* hero */
  A.push(`<div class="hero" style="grid-template-columns:repeat(auto-fit,minmax(180px,1fr))">
   <div class="tile"><div class="k">base-model acceptance</div><div class="v">${N(b0,3)}</div>
     <div class="d">shared drafter vs. unadapted target, &gamma;=4</div></div>
   <div class="tile"><div class="k">in-distribution tenant</div><div class="v" style="color:var(--series-2)">${N(bIn,3)}</div>
     <div class="d">${N(t0,2)} → ${N(tIn,2)} tokens/step</div></div>
   <div class="tile"><div class="k">held-out tenant</div><div class="v" style="color:var(--series-2)">${N(bOut,3)}</div>
     <div class="d">${N(t0,2)} → ${N(tOut,2)} tokens/step — speculation stops paying</div></div>
   <div class="tile"><div class="k">with companion adapter</div><div class="v" style="color:var(--series-3)">${N(comp?comp.beta:null,3)}</div>
     <div class="d">rank-4 LoRA on the drafter, ${comp?comp.params.toLocaleString():'—'} params</div></div>
   <div class="tile"><div class="k">companion vs. private drafter</div><div class="v">${D.e3.scenarios?Math.round(D.e3.scenarios[0].companion_vs_private_drafter)+'&times;':'—'}</div>
     <div class="d">smaller, per tenant (8B + 1B drafter)</div></div>
  </div>`);

  /* ---- E1 ---- */
  A.push(`<h2>1 &nbsp;The collapse</h2>
   <p class="note">Each tenant's LoRA is applied at a runtime strength <code>s</code> from 0 (base model) to 1.3.
   Evaluation sequences are regenerated from the adapted target at every point, so the drafter is always measured
   on the distribution it would actually see in deployment.</p>`);

  const tenants=[...new Set(D.e1.map(r=>r.tenant))];
  const sm=tenants.map(t=>{
    const rows=D.e1.filter(r=>r.tenant===t).sort((a,b)=>a.strength-b.strength);
    const kind=rows[0]?rows[0].kind:'';
    return `<div class="card"><h3>tenant ${t} &middot; ${esc(kind)}</h3>`+
      lineChart({series:[
        {name:'stochastic',color:'--series-1',points:rows.map(r=>({x:r.strength,y:r.beta_analytic,note:'shift '+N(r.shift_tvd,4)}))},
        {name:'greedy',color:'--series-2',points:rows.map(r=>({x:r.strength,y:r.beta_greedy,note:'shift '+N(r.shift_tvd,4)}))}],
        yDomain:[0,1.0],xLabel:'adapter strength',yLabel:'acceptance',width:440,height:200,
        xTicks:4,yTicks:4,directLabel:false})+
      `</div>`;});
  A.push(legend([{name:'stochastic acceptance  β = Σ min(p,q)',color:'--series-1'},
                 {name:'greedy acceptance  (top-1 agreement)',color:'--series-2'}]));
  A.push(`<div class="grid2">${sm.join('')}</div>`);
  A.push(`<p class="note"><b>Greedy decoding is hit far harder than stochastic.</b> Total-variation acceptance
   degrades gently because the accept/reject rule salvages probability mass wherever the two distributions still
   overlap; top-1 agreement has no such cushion, and most production serving runs at low temperature.</p>`);

  /* scatter: beta vs shift */
  const byKind={};
  D.e1.forEach(r=>{(byKind[r.kind]=byKind[r.kind]||[]).push(r)});
  const kcol={'in-dist':'--series-1','held-out':'--series-2'};
  const maxShift=Math.max(...D.e1.map(r=>r.shift_tvd));
  const slopesOf=f=>{const v=[];[...new Set(D.e1.map(r=>r.tenant))].forEach(t=>{
      const rs=D.e1.filter(r=>r.tenant===t).sort((a,b)=>a.strength-b.strength), z=rs[0];
      rs.slice(1).filter(f).forEach(r=>{if(r.shift_tvd>1e-6)v.push((z.beta_analytic-r.beta_analytic)/r.shift_tvd)});});
    return v.reduce((a,b)=>a+b,0)/v.length;};
  const slopeAll=slopesOf(()=>true), slopeLo=slopesOf(r=>r.strength<=0.45), slopeHi=slopesOf(r=>r.strength>=1.0);
  A.push(`<div class="card"><h3>acceptance against measured distribution shift</h3>`+
    legend(Object.keys(byKind).map(k=>({name:k+' tenants',color:kcol[k]||'--series-3'})).concat(
      [{name:'first-order prediction β₀ − Δ',color:'--neutral'}]))+
    scatter({series:Object.entries(byKind).map(([k,v])=>({name:k,color:kcol[k]||'--series-3',
        points:v.map(r=>({x:r.shift_tvd,y:r.beta_analytic,note:'tenant '+r.tenant+', s='+N(r.strength,2)}))})),
      xLabel:'Δ = TVD(base target, adapted target)',yLabel:'β',
      theory:{x0:0,y0:b0,x1:maxShift,y1:b0-maxShift,label:'β₀ − Δ  (triangle-inequality bound)'}})+
    `<p class="note">Measured slope of β against Δ: <b>${N(slopeAll,2)}</b> overall — ${N(slopeLo,2)} while the
    adapter is weak (s ≤ 0.45), ${N(slopeHi,2)} once it is at full strength (s ≥ 1.0). The points sit on or just
    above the triangle-inequality bound β₀ − Δ: the drafter absorbs part of the shift at first and stops being
    able to as the adapter strengthens. The relationship is linear throughout, as first-order theory predicts.</p>`+
    table(['tenant','kind','strength','shift Δ','β','greedy','tokens/step'],
      D.e1.map(r=>[r.tenant,r.kind,N(r.strength,2),N(r.shift_tvd,4),N(r.beta_analytic,4),
                   N(r.beta_greedy,4),N(r.tokens_per_step_iid,3)]),'t-e1')+`</div>`);

  /* ---- E2 ---- */
  if(D.e2.length){
    A.push(`<h2>2 &nbsp;The repair</h2>
     <p class="note">Every arm trains something on the drafter for one tenant, then measures acceptance on held-out
     rollouts from that tenant's adapted target, averaged over all six. <code>full-ft-tvd</code> is the
     memory-unbounded comparison — a whole private drafter per tenant, 26× the parameters. It wins, as it should;
     the claim is that 20 KB gets most of the way there. <code>companion-ce-r4</code> is the same adapter trained
     with hard-label cross-entropy on the tenant's tokens instead of distilling from the adapted target — it
     barely helps, which is the sharpest evidence that distillation, not fine-tuning, is doing the work.</p>`);
    const base=shared2?shared2.beta:0;
    const bars=D.e2.map(a=>({label:a.arm,value:a.beta,
      color:a.arm==='shared-drafter'?'--series-2':(a.arm==='full-ft-tvd'?'--neutral':'--series-3'),
      note:(a.params?a.params.toLocaleString()+' params · '+(a.bytes/1024).toFixed(0)+' KB fp16':'no extra params')}));
    A.push(`<div class="card"><h3>mean acceptance β by arm</h3>`+
      legend([{name:'shared drafter (baseline)',color:'--series-2'},{name:'companion adapter',color:'--series-3'},
              {name:'full drafter fine-tune (26× params)',color:'--neutral'}])+
      barChart({bars,min:0,max:Math.max(b0,...bars.map(b=>b.value))*1.02,
        valueLabel:'β',rule:b0,ruleLabel:'base-model β₀',labelW:190})+
      table(['arm','β','greedy','tokens/step','extra params','fp16 KB','train s'],
        D.e2.map(a=>[a.arm,N(a.beta,4),N(a.greedy,4),N(a.tok,3),a.params.toLocaleString(),
                     (a.bytes/1024).toFixed(1),N(a.train_s,1)]),'t-e2')+`</div>`);
  }

  /* ---- E4 ---- */
  if(D.e4.length){
    A.push(`<h2>3 &nbsp;QLoRA: train-time and serve-time quantisation</h2>
     <p class="note">Tenants fine-tune against a 4-bit base and hand the adapter to a server whose base may be
     quantised differently. <b>This one came out the other way.</b> The mismatch moves the served model
     meaningfully away from what the tenant trained (TVD up to 0.098 — see the table), but it costs the drafter
     almost no acceptance: round-to-nearest noise is near-isotropic in logit space, while a tenant adapter moves
     the target coherently in one direction. The tenant's LoRA is the problem; the quantisation mismatch is not.</p>`);
    A.push(`<div class="card"><h3>acceptance by quantisation arm</h3>`+
      legend([{name:'shared drafter',color:'--series-2'},{name:'+ companion adapter',color:'--series-3'}])+
      barChart({bars:D.e4.flatMap(r=>[
        {label:r.arm+' · shared',value:r.shared,color:'--series-2',note:'drift '+N(r.drift,4)},
        {label:r.arm+' · companion',value:r.companion,color:'--series-3',note:'drift '+N(r.drift,4)}]),
        min:0,max:1.0,valueLabel:'β',labelW:210,rowH:24})+
      table(['arm','drift from reference','β shared','β companion','tok/step shared','tok/step companion'],
        D.e4.map(r=>[r.arm,N(r.drift,4),N(r.shared,4),N(r.companion,4),N(r.tok_shared,3),N(r.tok_comp,3)]),'t-e4')+
      `</div>`);
  }

  /* ---- E3 ---- */
  if(D.e3.scenarios){
    A.push(`<h2>4 &nbsp;What it is worth in a server</h2>
     <p class="note">Measured acceptance pushed through the step-cost model. Decode is memory-bandwidth bound, so a
     forward pass costs roughly what its weights cost to read; the draft/target cost ratio follows the parameter
     ratio. Block drafters (DFlash-style) pay one drafter forward per block instead of &gamma;.</p>`);
    A.push(`<div class="card"><h3>speculative speedup, &gamma;=4</h3>`+
      legend([{name:'shared drafter',color:'--series-2'},{name:'+ companion adapter',color:'--series-3'},
              {name:'base-model ceiling',color:'--neutral'}])+
      barChart({bars:D.e3.scenarios.flatMap(s=>[
        {label:s.scenario.slice(0,34)+' · shared',value:s.speedup_shared,color:'--series-2'},
        {label:'· companion',value:s.speedup_companion,color:'--series-3',
         note:'recovers '+N(100*s.fraction_of_lost_speedup_recovered,1)+'% of lost speedup'},
        {label:'· ceiling',value:s.speedup_base_model_upper_bound,color:'--neutral'}]),
        min:0,valueLabel:'speedup',fmt:v=>v.toFixed(2)+'×',labelW:250,rowH:22})+
      table(['scenario','cost ratio','β shared','β companion','speedup shared','speedup companion','ceiling',
             'companion MB/tenant','private drafter GB/tenant'],
        D.e3.scenarios.map(s=>[s.scenario,N(s.cost_ratio,3),N(s.beta_shared,4),N(s.beta_companion,4),
          N(s.speedup_shared,3),N(s.speedup_companion,3),N(s.speedup_base_model_upper_bound,3),
          (s.companion_bytes_per_tenant/1e6).toFixed(2),
          (s.private_drafter_bytes_per_tenant/1e9).toFixed(2)]),'t-e3')+`</div>`);
  }

  document.getElementById('app').innerHTML=A.join('');
  document.querySelectorAll('.mk').forEach(el=>{
    el.addEventListener('mousemove',e=>showTip(e,el.dataset.tip));
    el.addEventListener('mouseleave',hideTip);});
  document.getElementById('foot').innerHTML=
    `CPU-scale pilot: sticky-HMM synthetic source, ${D.config.vocab||256}-token vocabulary; target
     ${(D.stage0.target_params/1e6||0).toFixed(2)}M params, drafter ${(D.stage0.draft_params/1e6||0).toFixed(2)}M
     (${(D.stage0.target_params/D.stage0.draft_params||0).toFixed(1)}× smaller); tenant LoRAs rank
     ${D.config.task_rank||8}, companion rank ${D.config.companion_rank||4}; γ=${D.config.gamma||4}, temperature
     ${D.config.temperature||1}. Acceptance simulator verified against the closed form and against a losslessness
     check (tests/test_accept.py). Numbers here are the pilot's; the T4 harness in experiments/gpu/ runs the same
     measurement code on Qwen2.5-1.5B + Qwen2.5-0.5B.`;
}
render();
</script></body></html>"""


def main():
    data = build_data()
    out = HTML.replace("__DATA__", json.dumps(data))
    p = f"{RES}/tandemspec_dashboard.html"
    open(p, "w").write(out)
    print(f"wrote {p} ({len(out)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
