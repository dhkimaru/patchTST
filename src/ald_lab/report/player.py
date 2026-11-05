"""Inference replay: a trained model's predictions on one run, played back in time with a button.

The report embeds the validation-run predictions each selected job already wrote
(``jobs/<id>/predictions_sample.parquet``); ``ald-predict --html`` embeds predictions it just made on
new trace CSVs. Both become the same *series* (plain dicts, JSON-ready) and the same player: a chart
with the model's input window, the predictions it makes at the current moment, the actual values as
they arrive, and the running error. Nothing here runs a model.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

MAX_POINTS = 20_000  # per view; longer runs are thinned evenly


SIGNIFICANT = 5  # digits kept per value: 753.68 Torr, 1.2345 Torr, 600.12 ℃


def _numbers(values: np.ndarray, digits: int | None = None) -> list[float | None]:
    """JSON-ready numbers: ``digits`` decimals, or :data:`SIGNIFICANT` significant digits per value."""

    if digits is not None:
        return [None if not np.isfinite(v) else round(float(v), digits) for v in values]
    return [None if not np.isfinite(v) else float(f"{v:.{SIGNIFICANT}g}") for v in values]


def _runs(labels: list[str]) -> list[list[Any]]:
    """Run-length list of [first index, label] for labels that change rarely (step names)."""

    out: list[list[Any]] = []
    for i, label in enumerate(labels):
        if not out or out[-1][1] != label:
            out.append([i, label])
    return out


def _keep(n: int) -> np.ndarray:
    return np.linspace(0, n - 1, MAX_POINTS).round().astype(int) if n > MAX_POINTS else np.arange(n)


def view(name: str, t: np.ndarray, actual: list[np.ndarray] | None, in_cycle: np.ndarray | None = None,
         steps: list[str] | None = None) -> dict[str, Any] | None:
    """One stretch of a run: times (s from the run start) and, per horizon, the actual values."""

    t = np.asarray(t, dtype=float)
    if len(t) < 3:
        return None
    keep = _keep(len(t))
    out: dict[str, Any] = {
        "name": name, "dt": round(float(np.median(np.diff(t[keep]))), 3), "t": _numbers(t[keep], 2),
        "y": [_numbers(np.asarray(a, dtype=float)[keep]) for a in actual] if actual else None,
    }
    if in_cycle is not None:
        out["cyc"] = "".join("1" if c else "0" for c in np.asarray(in_cycle, dtype=bool)[keep])
    if steps is not None:
        out["steps"] = _runs([str(steps[i]) for i in keep])
    return out


def preds(pred: list[np.ndarray]) -> list[list[float | None]]:
    """One model's predictions for a view (per horizon), thinned like :func:`view`."""

    keep = _keep(len(pred[0])) if pred else np.arange(0)
    return [_numbers(np.asarray(a, dtype=float)[keep]) for a in pred]


def series(key: str, *, group: str, title: str, target: str, unit: str, run: str, context: float,
           horizons: list[float], views: list[dict[str, Any] | None],
           models: list[dict[str, Any]]) -> dict[str, Any] | None:
    """A target on one run: shared views (time, actual) and one entry per model with ``p`` per view.
    ``models[i]`` is ``{"name", "label", "note", "p": [per view: [per horizon: values]]}``."""

    shown = [k for k, v in enumerate(views) if v]
    models = [{**m, "p": [m["p"][k] for k in shown]} for m in models if m.get("p")]
    if not shown or not models:
        return None
    return {"key": key, "group": group, "title": title, "target": target, "unit": unit, "run": run,
            "ctx": float(context), "h": [float(h) for h in horizons], "views": [views[k] for k in shown],
            "models": models, "default": 0}


def _sample_parts(frame: pd.DataFrame | None, zoom: pd.DataFrame | None) -> tuple[list[str], list[tuple[str, pd.DataFrame]]]:
    source = zoom if zoom is not None else frame
    if source is None:
        return [], []
    labels = [c[len("pred_"):] for c in source.columns if c.startswith("pred_")]
    parts = [(name, f) for f, name in ((zoom, "ALD 사이클 10분 확대"), (frame, "run 전체"))
             if f is not None and all(f"true_{x}" in f for x in labels)]
    return labels, parts


def sample_series(key: str, *, group: str, title: str, target: str, unit: str, context: float,
                  samples: list[dict[str, Any]]) -> dict[str, Any] | None:
    """A target from several jobs' ``predictions_sample.parquet`` of the same validation run (columns ``t_s``,
    ``in_cycle``, ``true_<h>s``, ``pred_<h>s``): views from the first, one model entry per job.
    ``samples[i]`` is ``{"name", "label", "note", "frame", "zoom"}`` (zoom = full resolution, frame = thinned run)."""

    if not samples:
        return None
    labels, parts = _sample_parts(samples[0]["frame"], samples[0]["zoom"])
    if not parts:
        return None
    views = [view(name, f["t_s"].to_numpy(float), [f[f"true_{x}"].to_numpy(float) for x in labels],
                  f["in_cycle"].to_numpy(bool) if "in_cycle" in f else None) for name, f in parts]
    models = []
    for sample in samples:
        labels2, parts2 = _sample_parts(sample["frame"], sample["zoom"])
        same = labels2 == labels and [n for n, _ in parts2] == [n for n, _ in parts] and all(
            len(f2) == len(f) and np.allclose(f2["t_s"].to_numpy(float), f["t_s"].to_numpy(float))
            for (_, f), (_, f2) in zip(parts, parts2))
        if same:  # every job predicts the same validation windows; a mismatch would mean another run
            models.append({"name": sample["name"], "label": sample["label"], "note": sample.get("note", ""),
                           "p": [preds([f2[f"pred_{x}"].to_numpy(float) for x in labels]) for _, f2 in parts2]})
    run = str(samples[0]["frame"]["run_id"].iloc[0]) if samples[0]["frame"] is not None else ""
    return series(key, group=group, title=title, target=target, unit=unit, run=run, context=context,
                  horizons=[float(x.rstrip("s")) for x in labels], views=views, models=models)


def prediction_series(frame: pd.DataFrame, *, model: str, context: float) -> list[dict[str, Any]]:
    """One series per run from ``ald-predict`` output (``pred`` or ``pred_+<h>s`` and the ``actual`` twins)."""

    cols = [c for c in frame.columns if c == "pred" or c.startswith("pred_+")]
    horizons = [0.0 if c == "pred" else float(c[len("pred_+"):].rstrip("s")) for c in cols]
    has_actual = all(c.replace("pred", "actual", 1) in frame for c in cols)
    target, unit = str(frame.attrs.get("target", "")), str(frame.attrs.get("unit", ""))
    out = []
    for run_id, group in frame.groupby("run_id", sort=False):
        stamps = pd.to_datetime(group["timestamp"], errors="coerce")
        t = (stamps - stamps.iloc[0]).dt.total_seconds().to_numpy(float)
        if not np.isfinite(t).all():
            t = np.arange(len(group), dtype=float)
        v = view("run 전체", t, [group[c.replace("pred", "actual", 1)].to_numpy(float) for c in cols] if has_actual
                 else None, group["in_cycle"].to_numpy(bool) if "in_cycle" in group else None,
                 group["step_name"].astype(str).tolist() if "step_name" in group else None)
        s = series(f"{run_id}", group="새 trace", title=f"{run_id}", target=target, unit=unit, run=str(run_id),
                   context=context, horizons=horizons, views=[v],
                   models=[{"name": model, "label": model, "note": "", "p": [preds([group[c].to_numpy(float)
                                                                                   for c in cols])]}])
        if s:
            out.append(s)
    return out


def player(items: list[dict[str, Any]]) -> str:
    """The player block (controls, chart, readout) with its data; :data:`PLAYER_JS` drives it."""

    data = json.dumps(items, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return (
        '<div class="player" id="player">'
        '<div class="pl-bar">'
        '<label class="pl-pick">대상 <select id="pl-series"></select></label>'
        '<span class="pl-views" id="pl-views"></span>'
        "</div>"
        '<div class="pl-bar" id="pl-mrow">모델 <span class="pl-views" id="pl-models"></span></div>'
        '<div class="pl-bar">'
        '<button type="button" class="pl-btn" id="pl-first" title="처음으로">⏮ 처음</button>'
        '<button type="button" class="pl-btn pl-main" id="pl-play">▶ 재생</button>'
        '<button type="button" class="pl-btn" id="pl-step" title="한 칸 앞으로">한 칸 ▶</button>'
        '<label>속도 <select id="pl-speed"><option value="1">1초에 1초</option><option value="5">×5</option>'
        '<option value="10" selected>×10</option><option value="30">×30</option><option value="120">×120</option>'
        "</select></label>"
        '<label id="pl-span-box">보이는 폭 <select id="pl-span"><option value="600">10분</option>'
        '<option value="1800">30분</option><option value="0">전체</option></select></label>'
        '<label class="pl-check"><input type="checkbox" id="pl-reveal"> 앞으로의 실측 미리 보기</label>'
        "</div>"
        '<input type="range" id="pl-slider" min="0" max="1" value="0" step="1" aria-label="재생 위치">'
        '<div class="pl-hz" id="pl-hz"></div>'
        '<div class="chartbox"><svg class="chart" id="pl-svg" viewBox="0 0 960 330" role="img" aria-label="추론 재생"></svg>'
        '<div class="legend" id="pl-legend"></div></div>'
        '<div class="pl-read" id="pl-read"></div>'
        f'<script type="application/json" id="pl-data">{data}</script>'
        "</div>"
    )


PLAYER_CSS = """
.player{margin:8px 0}
.pl-bar{display:flex;flex-wrap:wrap;align-items:center;gap:6px 10px;margin:6px 0;font-size:13px;color:var(--text2)}
.pl-bar select{font:inherit;color:var(--text);background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:4px 6px;max-width:100%}
.pl-pick select{min-width:260px}
.pl-btn{font:inherit;border:1px solid var(--line);background:var(--surface);color:var(--text);border-radius:8px;padding:5px 12px;cursor:pointer}
.pl-btn.pl-main{background:var(--s1);border-color:var(--s1);color:#fff;font-weight:700;min-width:96px}
.pl-btn.on{background:var(--text);color:var(--page);border-color:var(--text)}
.pl-views{display:flex;gap:4px;flex-wrap:wrap}
.pl-check{display:flex;align-items:center;gap:4px}
#pl-slider{width:100%;accent-color:var(--s1);margin:2px 0 4px}
.player .chartbox{overflow-x:auto} #pl-svg{min-width:640px}
.pl-hz{display:flex;flex-wrap:wrap;gap:4px;align-items:center;font-size:12px;color:var(--text2)}
.pl-read{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(210px,100%),1fr));gap:8px;margin:6px 0}
.pl-card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:8px 10px;font-size:12px}
.pl-card b{display:block;font-size:13px;color:var(--text)} .pl-card .big{font-size:20px;font-weight:700;color:var(--text)}
.pl-card .row{display:flex;justify-content:space-between;gap:8px} .pl-card .k{color:var(--muted)}
#pl-svg .ln-true{stroke:var(--muted);stroke-width:2.6;opacity:.9} #pl-svg .ln-pred{stroke-width:1.4}
svg .pl-win{fill:var(--s1);fill-opacity:.10;stroke:var(--s1);stroke-opacity:.5;stroke-dasharray:3 3}
svg .pl-cyc{fill:var(--shade)} svg .pl-now{stroke:var(--text);stroke-width:1.2}
svg .pl-future{fill:none;stroke:var(--text2);stroke-width:1.1;stroke-dasharray:3 4;opacity:.55}
svg .pl-fan{stroke:var(--s3);stroke-width:1.2;stroke-dasharray:2 3} svg .pl-dot{fill:var(--s3);stroke:var(--surface);stroke-width:1.5}
svg .pl-ok{fill:var(--s1);stroke:var(--surface);stroke-width:1.5}
svg .pl-lbl{fill:var(--text);font-size:11.5px;font-weight:600;paint-order:stroke;stroke:var(--surface);stroke-width:4px}
.legend i.pl-l-win{background:color-mix(in srgb,var(--s1) 14%,transparent);border:1px dashed var(--s1)}
.legend i.pl-l-dot{background:var(--s3);border-radius:50%}
@media print{.pl-bar,#pl-slider{display:none}}
"""

PLAYER_JS = r"""
(function(){
const root=document.getElementById('player'); if(!root) return;
const ITEMS=JSON.parse(document.getElementById('pl-data').textContent);
const $=id=>document.getElementById(id);
const sel=$('pl-series'), svg=$('pl-svg'), slider=$('pl-slider'), playBtn=$('pl-play');
const W=960,H=330,L=62,R=16,T=18,B=40;
let S=null,V=null,P=null,vi=0,mi=0,i=0,hsel=0,playing=false,last=0,acc=0,A=null;
const groups={};ITEMS.forEach((s,k)=>{(groups[s.group]=groups[s.group]||[]).push(k);});
Object.keys(groups).forEach(g=>{const og=document.createElement('optgroup');og.label=g;
  groups[g].forEach(k=>{const o=document.createElement('option');o.value=k;o.textContent=g+' · '+ITEMS[k].title;og.appendChild(o);});sel.appendChild(og);});
function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
function fmtT(s){s=Math.max(0,Math.round(s));const h=Math.floor(s/3600),m=Math.floor(s%3600/60),x=s%60;
  return (h?h+':'+String(m).padStart(2,'0'):m)+':'+String(x).padStart(2,'0');}
function nice(v){if(v===null||!isFinite(v))return '-';const a=Math.abs(v);
  return a>=1000?v.toFixed(0):a>=100?v.toFixed(1):a>=10?v.toFixed(2):a>=1?v.toFixed(3):v.toPrecision(3);}
function last_le(arr,x){let lo=0,hi=arr.length-1,r=-1;while(lo<=hi){const m=(lo+hi)>>1;if(arr[m]<=x){r=m;lo=m+1;}else hi=m-1;}return r;}
function prepare(){
  V=S.views[vi]; P=S.models[mi].p[vi]; const n=V.t.length, h=S.h;
  // actual series on its own time axis: value at t+h measured; smallest horizon first, the rest extend the tail
  const at=[],av=[];
  if(V.y){const order=h.map((x,j)=>j).sort((a,b)=>h[a]-h[b]);
    order.forEach(j=>{for(let k=0;k<n;k++){const tt=V.t[k]+h[j],v=V.y[j][k];if(v===null)continue;
      if(!at.length||tt>at[at.length-1]){at.push(tt);av.push(v);}}});}
  // cumulative absolute error per horizon, in the order predictions are checked (t+h)
  const cum=h.map((x,j)=>{const c=new Float64Array(n+1);let cnt=new Int32Array(n+1);
    for(let k=0;k<n;k++){const p=P[j][k],y=V.y?V.y[j][k]:null;const ok=p!==null&&y!==null;
      c[k+1]=c[k]+(ok?Math.abs(p-y):0);cnt[k+1]=cnt[k]+(ok?1:0);}return {c:c,n:cnt};});
  const cyc=[]; if(V.cyc){let s=-1;for(let k=0;k<=n;k++){const on=k<n&&V.cyc[k]==='1';
    if(on&&s<0)s=k; if(!on&&s>=0){cyc.push([V.t[s],V.t[k-1]]);s=-1;}}}
  const all=V.cyc&&V.cyc.indexOf('0')<0;
  A={at:at,av:av,cum:cum,cyc:all?[]:cyc,tend:Math.max(V.t[n-1]+Math.max(...h),at.length?at[at.length-1]:0)};
  slider.max=n-1; i=Math.min(i,n-1);
}
function ticks(lo,hi,n){const span=hi-lo,step0=span/n,mag=Math.pow(10,Math.floor(Math.log10(step0)));
  const step=[1,2,2.5,5,10].map(x=>x*mag).find(x=>span/x<=n)||mag*10;const out=[];
  for(let v=Math.ceil(lo/step)*step;v<=hi+1e-9;v+=step)out.push(+v.toFixed(10));out.step=step;return out;}
function tickFmt(v,step){let d=0;while(d<5&&Math.abs(step*Math.pow(10,d)-Math.round(step*Math.pow(10,d)))>1e-6)d++;return v.toFixed(d);}
function yrange(v0,v1,t){let lo=Infinity,hi=-Infinity;const upd=v=>{if(v!==null){if(v<lo)lo=v;if(v>hi)hi=v;}};
  const a=Math.max(0,last_le(A.at,v0)),b=last_le(A.at,v1);for(let k=a;k<=b;k++)upd(A.av[k]);
  const c=Math.max(0,last_le(V.t,v0-Math.max(...S.h))),d=last_le(V.t,v1);
  for(let k=c;k<=d;k++)P.forEach(arr=>upd(arr[k]));
  if(!isFinite(lo)){lo=0;hi=1;} if(hi-lo<1e-9){hi+=1;lo-=1;} const pad=(hi-lo)*0.07;return [lo-pad,hi+pad];}
function draw(){
  if(!V)return; const n=V.t.length, t=V.t[i], h=S.h, span=+$('pl-span').value;
  const t0=V.t[0], tEnd=A.tend; let v0=t0,v1=tEnd;
  if(span>0&&tEnd-t0>span){v0=Math.min(Math.max(t0,t-span*0.62),tEnd-span);v1=v0+span;}
  const yr=yrange(v0,v1,t);A.lo=yr[0];A.hi=yr[1];
  const X=x=>L+(x-v0)/(v1-v0)*(W-L-R), Y=y=>T+(A.hi-y)/(A.hi-A.lo)*(H-T-B);
  const o=[];
  A.cyc.forEach(c=>{if(c[1]<v0||c[0]>v1)return;const a=X(Math.max(c[0],v0)),b=X(Math.min(c[1],v1));
    o.push(`<rect class="pl-cyc" x="${a.toFixed(1)}" y="${T}" width="${Math.max(1,b-a).toFixed(1)}" height="${H-T-B}"/>`);});
  const yt=ticks(A.lo,A.hi,5);yt.forEach(v=>{const y=Y(v);o.push(`<line class="grid" x1="${L}" x2="${W-R}" y1="${y.toFixed(1)}" y2="${y.toFixed(1)}"/>`,
    `<text class="tick" x="${L-6}" y="${(y+4).toFixed(1)}" text-anchor="end">${tickFmt(v,yt.step)}</text>`);});
  const tk=ticks(v0,v1,7);tk.forEach(v=>{const x=X(v);o.push(`<text class="tick" x="${x.toFixed(1)}" y="${H-B+16}" text-anchor="middle">${fmtT(v)}</text>`);});
  o.push(`<line class="axis" x1="${L}" x2="${W-R}" y1="${H-B}" y2="${H-B}"/>`);
  o.push(`<text class="tick" x="${W-R}" y="${H-6}" text-anchor="end">run 시작 후 시간 (분:초)</text>`);
  if(S.unit)o.push(`<text class="tick" x="${L-6}" y="${T-6}" text-anchor="end">${esc(S.unit)}</text>`);
  // the model's input window
  const w0=Math.max(v0,t-S.ctx);o.push(`<rect class="pl-win" x="${X(w0).toFixed(1)}" y="${T}" width="${Math.max(1,X(t)-X(w0)).toFixed(1)}" height="${H-T-B}"/>`);
  if(X(t)-X(w0)>70)o.push(`<text class="tick" x="${((X(w0)+X(t))/2).toFixed(1)}" y="${T+12}" text-anchor="middle">모델 입력 (최근 ${S.ctx}초)</text>`);
  // thinned polylines inside the view
  function line(ts,vs,cls,from,to){const a=Math.max(0,last_le(ts,from)),b=last_le(ts,to);if(b<a)return;
    const stp=Math.max(1,Math.floor((b-a)/1400));const pts=[];
    for(let k=a;k<=b;k+=stp){if(vs[k]!==null&&ts[k]>=v0)pts.push(X(ts[k]).toFixed(1)+','+Y(vs[k]).toFixed(1));}
    if(pts.length>1)o.push(`<polyline class="${cls}" points="${pts.join(' ')}"/>`);}
  if(V.y){line(A.at,A.av,'ln-true',v0,t); if($('pl-reveal').checked)line(A.at,A.av,'pl-future',t,v1);}
  const hj=Math.min(hsel,h.length-1);
  const pt=V.t.map(x=>x+h[hj]); line(pt,P[hj],'ln-pred',v0,t+h[hj]);
  // the predictions made now
  const ka=last_le(A.at,t); const ya=ka>=0&&A.at[ka]>t-2*Math.max(V.dt,1)?A.av[ka]:null;
  const labels=[];
  h.forEach((hh,j)=>{const p=P[j][i];if(p===null)return;const x=X(t+hh),y=Y(p);
    if(hh>0&&ya!==null)o.push(`<line class="pl-fan" x1="${X(t).toFixed(1)}" y1="${Y(ya).toFixed(1)}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}"/>`);
    o.push(`<circle class="pl-dot" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="5"/>`);
    labels.push([x+7,y-7,hh>0?'+'+hh+'초':'추정']);});
  labels.sort((a,b)=>b[1]-a[1]);
  for(let k=1;k<labels.length;k++){const pr=labels[k-1];if(pr[1]-labels[k][1]<14&&Math.abs(pr[0]-labels[k][0])<48)labels[k][1]=pr[1]-14;}
  labels.forEach(l=>o.push(`<text class="pl-lbl" x="${l[0].toFixed(1)}" y="${Math.max(T+10,l[1]).toFixed(1)}">${l[2]}</text>`));
  if(ya!==null&&h[0]>0)o.push(`<circle class="pl-ok" cx="${X(t).toFixed(1)}" cy="${Y(ya).toFixed(1)}" r="4"/>`);
  o.push(`<line class="pl-now" x1="${X(t).toFixed(1)}" x2="${X(t).toFixed(1)}" y1="${T}" y2="${H-B}"/>`,
         `<text class="pl-lbl" x="${X(t).toFixed(1)}" y="${H-B-6}" text-anchor="middle">지금 ${fmtT(t)}</text>`);
  svg.innerHTML=o.join('');
  slider.value=i; readout(t);
}
function stepName(k){if(!V.steps)return '';let s='';for(const r of V.steps){if(r[0]<=k)s=r[1];else break;}return s;}
function readout(t){
  const h=S.h,cards=[],cyc=V.cyc?V.cyc[i]==='1':null,step=stepName(i);
  cards.push(`<div class="pl-card"><b>지금</b><div class="big">${fmtT(t)}</div><div class="k">${esc(S.run)}`
    +(step?' · '+esc(step):'')+(cyc===null?'':cyc?' · ALD 사이클':' · 사이클 밖')+`</div></div>`);
  h.forEach((hh,j)=>{const p=P[j][i];
    // the newest prediction for this horizon whose answer has arrived (made hh seconds ago)
    const k=last_le(V.t,t-hh); let made='',err='';
    if(V.y&&k>=0){const pk=P[j][k],yk=V.y[j][k];const c=A.cum[j];
      if(pk!==null&&yk!==null)made=`<div class="row"><span class="k">${hh>0?hh+'초 전 예측':'추정'}</span><span>${nice(pk)}</span></div>`
        +`<div class="row"><span class="k">실측</span><span>${nice(yk)}</span></div>`
        +`<div class="row"><span class="k">오차</span><span>${nice(Math.abs(pk-yk))}</span></div>`;
      const cnt=c.n[k+1];if(cnt)err=`<div class="row"><span class="k">지금까지 MAE (${cnt}개)</span><span><b style="display:inline">${nice(c.c[k+1]/cnt)}</b></span></div>`;}
    const head=hh>0?`+${hh}초 뒤 예측`:'지금 값 추정';
    cards.push(`<div class="pl-card"><b>${head}</b><div class="big">${nice(p)} <span class="k" style="font-size:12px">${esc(S.unit)}</span></div>${made}${err}</div>`);});
  $('pl-read').innerHTML=cards.join('');
}
function legend(){const hj=Math.min(hsel,S.h.length-1),f=S.h[0]>0;
  $('pl-legend').innerHTML=(V.y?'<span><b class="ln" style="border-top:2.6px solid var(--muted)"></b>실측</span>':'')
   +`<span><b class="ln" style="border-top-color:var(--s1)"></b>${f?'+'+S.h[hj]+'초 예측 기록 (해당 시각의 예측값)':'추정 기록'}</span>`
   +`<span><i class="pl-l-dot"></i>${f?'현재 시점 예측':'지금 추정'}</span><span><i class="pl-l-win"></i>모델 입력 구간</span>`
   +(A.cyc.length?'<span><i style="background:var(--shade)"></i>ALD 사이클 구간</span>':'')
   +`<span class="sub">${esc(S.models[mi].label)} · ${esc(S.run)} · ${esc(V.name)}</span>`;}
function pick(k){S=ITEMS[k];vi=0;i=0;hsel=0;mi=S.default||0;stop();
  models();
  $('pl-views').innerHTML=S.views.length>1?S.views.map((v,j)=>`<button type="button" class="pl-btn${j===vi?' on':''}" data-v="${j}">${esc(v.name)}</button>`).join(''):'';
  $('pl-views').querySelectorAll('button').forEach(b=>b.onclick=()=>{vi=+b.dataset.v;i=0;stop();
    $('pl-views').querySelectorAll('button').forEach(x=>x.classList.toggle('on',x===b));setup();});
  $('pl-hz').innerHTML=S.h.length>1?'예측 기록선: '+S.h.map((hh,j)=>`<button type="button" class="pl-btn${j===hsel?' on':''}" data-h="${j}">+${hh}초</button>`).join(''):'';
  $('pl-hz').querySelectorAll('button').forEach(b=>b.onclick=()=>{hsel=+b.dataset.h;
    $('pl-hz').querySelectorAll('button').forEach(x=>x.classList.toggle('on',x===b));legend();draw();});
  setup();}
function models(){const box=$('pl-models');$('pl-mrow').style.display=S.models.length>1?'':'none';
  box.innerHTML=S.models.map((m,j)=>`<button type="button" class="pl-btn${j===mi?' on':''}" data-m="${j}" title="${esc(m.note||'')}">${esc(m.label)}</button>`).join('');
  box.querySelectorAll('button').forEach(b=>b.onclick=()=>{mi=+b.dataset.m;
    box.querySelectorAll('button').forEach(x=>x.classList.toggle('on',x===b));prepare();legend();draw();});}
function setup(){prepare();const long=A.tend-V.t[0]>700;$('pl-span-box').style.display=long?'':'none';
  if(!long)$('pl-span').value='0'; else if($('pl-span').value==='0')$('pl-span').value='600';
  i=Math.max(0,last_le(V.t,V.t[0]+Math.min(S.ctx,0.3*(A.tend-V.t[0])))); legend();draw();}
function stop(){playing=false;playBtn.textContent='▶ 재생';}
function frame(ts){if(!playing)return;const dt=Math.min(0.25,(ts-last)/1000);last=ts;
  acc+=dt*(+$('pl-speed').value);const target=V.t[i]+acc;let k=last_le(V.t,target);
  if(k>i){acc=target-V.t[k];i=k;draw();}
  if(i>=V.t.length-1){stop();return;} requestAnimationFrame(frame);}
playBtn.onclick=()=>{if(playing){stop();return;} if(i>=V.t.length-1)i=0;playing=true;acc=0;last=performance.now();
  playBtn.textContent='⏸ 일시정지';requestAnimationFrame(frame);};
$('pl-first').onclick=()=>{stop();setup();};
$('pl-step').onclick=()=>{stop();i=Math.min(i+1,V.t.length-1);draw();};
slider.oninput=()=>{i=+slider.value;acc=0;draw();};
sel.onchange=()=>pick(+sel.value); $('pl-reveal').onchange=draw; $('pl-span').onchange=draw;
if(ITEMS.length)pick(0);
})();
"""
