#!/usr/bin/env python3
"""IBKR 账户看板（数据源：PortfolioAnalyst 自开户报告）。
用法:
  python dashboard.py --ingest report_since_inception.plain      # 导入新快照并生成
  python dashboard.py                                            # 用库里最新快照重新生成
所有计算都在 pa_analytics 模块里，这里只负责渲染。"""
import argparse, json
from datetime import datetime
from pathlib import Path
from pa_analytics import Store, WINDOWS, WINDOW_LABEL, window_summary, window_attribution, yearly_table, get_capabilities

HTML = r"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>IBKR 账户看板（PA）</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#f5f4f1;--card:#fff;--border:rgba(0,0,0,.08);--text:#1a1a18;--muted:#6b6b67;--amber-bg:#faeeda;--amber-text:#633806;--amber-bd:#c8821a;--info:#1a5fa8;--info-bg:#edf4ff;--info-bd:#b3d0f5;--tooltip-bg:#1c1c1a;--tooltip-text:#e8e6df;--pos:#1a7f4b;--neg:#c0392b;--line:#1a5fa8}
@media(prefers-color-scheme:dark){:root{--bg:#1a1a18;--card:#242422;--border:rgba(255,255,255,.09);--text:#e8e6df;--muted:#9c9a92;--amber-bg:#412402;--amber-text:#fac775;--amber-bd:#ba7517;--info:#85b7eb;--info-bg:#1a2d44;--info-bd:#2d4d70;--tooltip-bg:#f0ede6;--tooltip-text:#1a1a18;--pos:#4cc38a;--neg:#ee7b6e;--line:#85b7eb}}
body{background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,sans-serif;font-size:14px}
.page{max-width:1200px;margin:0 auto;padding:20px 16px}
h1{font-size:18px;font-weight:500}.sub{font-size:12px;color:var(--muted);margin:3px 0 16px}
.tabs{display:flex;gap:2px;border-bottom:1px solid var(--border);margin-bottom:16px;overflow-x:auto}
.tab{padding:8px 14px;font-size:13px;font-weight:500;color:var(--muted);cursor:pointer;border:none;background:none;border-bottom:2px solid transparent;margin-bottom:-1px;white-space:nowrap}
.tab.active{color:var(--text);border-bottom-color:var(--text)}.tab:hover:not(.active){color:var(--text)}
.panel{display:none}.panel.active{display:block}
.g3{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:12px}.g1{margin-bottom:12px}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media(max-width:680px){.g3,.g2{grid-template-columns:1fr 1fr}.g2.stack{grid-template-columns:1fr}}
.cc{background:var(--card);border:.5px solid var(--border);border-radius:10px;position:relative}
.cc-head{padding:12px 14px 0;display:flex;align-items:flex-start;gap:8px}.cc-title{font-size:13px;font-weight:500;flex:1}
.cc-body{padding:10px 14px 14px}
.kpi .label{font-size:12px;color:var(--muted)}.kpi .val{font-size:22px;font-weight:500;margin-top:4px;font-variant-numeric:tabular-nums}.kpi .note{font-size:11px;color:var(--muted);margin-top:2px}
.pos{color:var(--pos)}.neg{color:var(--neg)}.muted{color:var(--muted)}
.info-btn{flex-shrink:0;width:18px;height:18px;border-radius:50%;background:var(--info-bg);border:1px solid var(--info-bd);color:var(--info);font-size:11px;font-weight:700;cursor:pointer;display:flex;align-items:center;justify-content:center;position:relative;user-select:none}
.tooltip{display:none;position:absolute;z-index:999;right:0;top:24px;width:290px;padding:12px 14px;background:var(--tooltip-bg);color:var(--tooltip-text);border-radius:8px;font-size:12px;line-height:1.6;font-weight:400;box-shadow:0 4px 20px rgba(0,0,0,.25);pointer-events:none;text-align:left}
.info-btn:hover .tooltip,.info-btn:focus .tooltip{display:block}.kpi .info-btn{position:absolute;top:10px;right:10px}
.banner{background:var(--amber-bg);border:.5px solid var(--amber-bd);border-radius:8px;padding:10px 14px;font-size:12px;color:var(--amber-text);margin-bottom:12px;line-height:1.6}
.fine{font-size:11px;color:var(--muted);margin:6px 0;line-height:1.6}
.bars .row{display:grid;grid-template-columns:200px 1fr 90px;gap:8px;align-items:center;padding:5px 0;font-size:12px}
@media(max-width:680px){.bars .row{grid-template-columns:130px 1fr 80px}}
.bars .track{height:10px;background:var(--border);border-radius:5px;position:relative}.bars .fill{position:absolute;top:0;bottom:0;left:0;border-radius:5px}
.bars .num{text-align:right;font-variant-numeric:tabular-nums}
.badge{font-size:10px;padding:1px 6px;border-radius:8px;border:.5px solid var(--border);color:var(--muted);margin-left:6px}
.tbl{width:100%;border-collapse:collapse;font-size:12px}.tbl th{font-weight:500;color:var(--muted);text-align:right;padding:6px 8px;border-bottom:1px solid var(--border)}
.tbl th:first-child,.tbl td:first-child{text-align:left}.tbl td{padding:6px 8px;border-bottom:.5px solid var(--border);text-align:right;font-variant-numeric:tabular-nums}
.chart{position:relative}.chart svg{width:100%;height:auto;display:block}
.hover{position:absolute;pointer-events:none;background:var(--tooltip-bg);color:var(--tooltip-text);font-size:11px;padding:5px 8px;border-radius:6px;display:none;white-space:nowrap;line-height:1.5}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:11px;color:var(--muted);margin-bottom:6px}.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
.k{padding:8px 0;border-bottom:.5px solid var(--border);font-size:12px;line-height:1.6}.k b{font-weight:500}
details summary{cursor:pointer;font-size:12px;color:var(--muted)}
</style></head><body><div class="page">
<h1>IBKR 账户看板</h1><div class="sub" id="sub"></div>
<div id="banners"></div><div class="tabs" id="tabs"></div><div id="panels"></div>
</div>
<script>
const D = __DATA__;
const CCY = D.currency, BM = D.benchmarks;
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fmt = (n, dp=0) => n==null ? '—' : Number(n).toLocaleString('en-US',{minimumFractionDigits:dp,maximumFractionDigits:dp});
const sgn = (n, dp=0) => n==null ? '—' : (n>0?'+':'')+fmt(n,dp);
const pc = (n, dp=2) => n==null ? '—' : sgn(n,dp)+'%';
const cls = n => n>0?'pos':(n<0?'neg':'');
const COL = ['var(--line)','#c8821a','#7a6ab8','#2a9d8f'];
const F = {W:800,H:250,L:62,R:14,T:12,B:26};
const kpi = (label,val,note,c='',tip='') => `<div class="cc kpi"><div class="cc-body"><div class="label">${label}</div><div class="val ${c}">${val}</div><div class="note">${note||'&nbsp;'}</div></div>${tip?`<div class="info-btn" tabindex="0">ⓘ<div class="tooltip">${tip}</div></div>`:''}</div>`;
const card = (title, id, tip='') => `<div class="cc g1"><div class="cc-head"><div class="cc-title">${title}</div>${tip?`<div class="info-btn" tabindex="0">ⓘ<div class="tooltip">${tip}</div></div>`:''}</div><div class="cc-body"><div class="chart" id="${id}"></div></div></div>`;

function shell(el, labels, x, gridLines, body, legend, n, render){
  let ax=''; for(let k=0;k<=4;k++){const i=Math.round((n-1)*k/4); ax+=`<text x="${x(i)}" y="${F.H-8}" text-anchor="${k==0?'start':k==4?'end':'middle'}" font-size="11" fill="currentColor" opacity=".6">${labels[i]}</text>`;}
  el.innerHTML=(legend||'')+`<svg viewBox="0 0 ${F.W} ${F.H}" preserveAspectRatio="none"><g style="color:var(--text)">${gridLines}${ax}</g>${body}<line class="cur" y1="${F.T}" y2="${F.H-F.B}" stroke="currentColor" opacity=".35" style="display:none"/></svg><div class="hover"></div>`;
  const svg=el.querySelector('svg'),tip=el.querySelector('.hover'),cur=svg.querySelector('.cur');
  svg.addEventListener('mousemove',e=>{const r=svg.getBoundingClientRect();const px=(e.clientX-r.left)/r.width*F.W;
    let i=Math.floor((px-F.L)/((F.W-F.L-F.R)/n)); i=Math.max(0,Math.min(n-1,i)); const xx=x(i);
    cur.setAttribute('x1',xx);cur.setAttribute('x2',xx);cur.style.display='';tip.style.display='block';tip.innerHTML=render(i);
    tip.style.left=Math.min(Math.max(0,xx/F.W*r.width+12),r.width-200)+'px';tip.style.top='4px';});
  svg.addEventListener('mouseleave',()=>{tip.style.display='none';cur.style.display='none';});
}
function grid(lo,hi,y,fy){let g='';for(let k=0;k<=4;k++){const v=lo+(hi-lo)*k/4,yy=y(v);g+=`<line x1="${F.L}" x2="${F.W-F.R}" y1="${yy}" y2="${yy}" stroke="currentColor" opacity=".1"/><text x="${F.L-6}" y="${yy+4}" text-anchor="end" font-size="11" fill="currentColor" opacity=".6">${fy(v)}</text>`;}return g;}
function lineChart(el, series, fy, flows){
  const n=series[0].pts.length, labels=series[0].pts.map(p=>p[0]);
  const vals=series.flatMap(s=>s.pts.map(p=>p[1])); let lo=Math.min(...vals),hi=Math.max(...vals); const pad=(hi-lo)*.08||1; lo-=pad; hi+=pad;
  const x=i=>F.L+(F.W-F.L-F.R)*(i+.5)/n, y=v=>F.T+(F.H-F.T-F.B)*(1-(v-lo)/(hi-lo));
  let body=series.map(s=>`<polyline points="${s.pts.map((p,i)=>x(i).toFixed(1)+','+y(p[1]).toFixed(1)).join(' ')}" fill="none" stroke="${s.color}" stroke-width="${s.w||1.8}" stroke-linejoin="round"/>`).join('');
  (flows||[]).forEach(f=>{let i=labels.findIndex(d=>d>=f[0]); if(i<0)i=n-1; body+=`<rect x="${x(i)-1.5}" y="${F.H-F.B-8}" width="3" height="8" fill="${f[1]>0?'var(--pos)':'var(--neg)'}" opacity=".85"><title>${f[0]} ${sgn(f[1])}</title></rect>`;});
  const legend=series.length>1?`<div class="legend">${series.map(s=>`<span><i style="background:${s.color}"></i>${esc(s.name)}</span>`).join('')}</div>`:'';
  shell(el,labels,x,grid(lo,hi,y,fy),body,legend,n,i=>`${labels[i]}<br>`+series.map(s=>`${esc(s.name)}: ${fy(s.pts[i][1],1)}`).join('<br>'));
}
const AK=[['etfs','ETF','#1a5fa8'],['stocks','股票','#2a9d8f'],['options','股票期权','#c8821a'],['futures_options','期货期权','#c0392b'],['cash','现金','#9c9a92']];
function stackChart(el, rows){
  const n=rows.length, labels=rows.map(r=>r.date);
  let hi=0,lo=0; rows.forEach(r=>{let p=0,q=0;AK.forEach(([k])=>{(r[k]||0)>=0?p+=r[k]:q+=r[k];});hi=Math.max(hi,p);lo=Math.min(lo,q);});
  const pad=(hi-lo)*.06||1; hi+=pad; if(lo<0)lo-=pad;
  const x=i=>F.L+(F.W-F.L-F.R)*(i+.5)/n, y=v=>F.T+(F.H-F.T-F.B)*(1-(v-lo)/(hi-lo)), bw=Math.max(1,(F.W-F.L-F.R)/n*.72);
  let body='';
  rows.forEach((r,i)=>{let p=0,q=0;AK.forEach(([k,,c])=>{const v=r[k]||0;if(!v)return;const a=v>=0?p:q,b=a+v;body+=`<rect x="${(x(i)-bw/2).toFixed(1)}" y="${Math.min(y(a),y(b)).toFixed(1)}" width="${bw.toFixed(1)}" height="${Math.abs(y(a)-y(b)).toFixed(1)}" fill="${c}" opacity=".85"/>`;v>=0?p=b:q=b;});});
  body+=`<line x1="${F.L}" x2="${F.W-F.R}" y1="${y(0)}" y2="${y(0)}" stroke="currentColor" opacity=".3"/>`;
  const legend=`<div class="legend">${AK.map(([,nm,c])=>`<span><i style="background:${c}"></i>${nm}</span>`).join('')}</div>`;
  shell(el,labels,x,grid(lo,hi,y,v=>fmt(v/1000)+'k'),body,legend,n,i=>`${labels[i]}<br>`+AK.map(([k,nm])=>`${nm}: ${fmt(rows[i][k])}`).join('<br>')+`<br>净值: ${fmt(rows[i].nav)}`);
}
const QL={exact:'精确',estimate:'估算(已验证)',unverified:'估算(未验证)',residual:'合并项'};
function attrBars(a){
  const items=a.buckets.map(b=>({label:b.label,amount:b.amount,q:QL[b.quality]+((!a.validated&&b.quality==='residual')?'·未验证':'')})), mx=Math.max(...items.map(i=>Math.abs(i.amount)),1);
  const total=items.reduce((s,i)=>s+i.amount,0);
  let h='<div class="bars">'+items.map(i=>`<div class="row"><div>${esc(i.label)}<span class="badge">${i.q}</span></div><div class="track"><div class="fill" style="width:${Math.abs(i.amount)/mx*100}%;background:${i.amount>=0?'var(--pos)':'var(--neg)'};opacity:.85"></div></div><div class="num ${cls(i.amount)}">${sgn(i.amount)}</div></div>`).join('')+'</div>';
  h+=`<div class="fine">合计 ${sgn(total)}（= 净值口径区间盈亏 ${sgn(a.profit)}） · 股息 ${sgn(a.memo_dividends)}（备注，归属未验证，已含在某一项内）</div>`;
  const r=a.pa_split_reference;
  h+=`<details><summary>PA 口径的期货/期货期权/现金拆分（仅供参考，不可靠，见 K3）</summary><div class="fine">期货 ${sgn(r.futures)} · 期货期权 ${sgn(r.futures_options)} · 类期货期权 ${sgn(r.future_style_options)} · 现金 ${sgn(r.cash)}（现金含利息、费用、外汇）</div></details>`;
  return h;
}
function winPanel(w){
  const S=D.windows[w].summary, A=D.windows[w].attribution;
  if(S.status!=='ok') return `<div class="banner">${esc(S.warnings.join('；'))}</div>`;
  const v=S.value;
  const off=v.official_pct!=null?`${esc(v.official_note)} ${pc(v.official_pct)}`:'月度 TWR 链接';
  const tip='<strong>TWR（时间加权）</strong>把 PA 的月度收益逐月链接，不受入金时点影响。窗口按月份标签对齐，含当月至今。官方的 1年/3年/5年 按日滚动，起点不同，所以会有差异。';
  let h=`<div class="g3">`+kpi('期初资产 '+v.start_date,fmt(v.start_nav),CCY)+kpi('转入',fmt(v.deposits),CCY)+kpi('转出',fmt(Math.abs(v.withdrawals)),CCY)+
    kpi(w==='YTD'?'今年以来收益':'区间收益',sgn(v.profit),CCY+' · 已扣除转入转出',cls(v.profit))+kpi('期末资产 '+v.end_date,fmt(v.end_nav),CCY)+
    kpi('收益率（TWR）',pc(v.twr_pct),off,cls(v.twr_pct),tip)+`</div>`;
  h+=`<div class="fine">${S.warnings.map(esc).join('；')}</div>`;
  h+=card(`资产曲线（月末，${CCY}）`,`nav-${w}`)+card('收益曲线：累计 TWR vs 基准',`ret-${w}`)+card('资产组合变化（月末市值，按品种；负值 = 卖出期权的负市值）',`alc-${w}`,'这是市值口径，不是风险口径：卖出期权的名义敞口没有体现，现金占比高不代表风险低（K9）。');
  h+=`<div class="cc g1"><div class="cc-head"><div class="cc-title">按资产类别归因（SGD）</div><div class="info-btn" tabindex="0">ⓘ<div class="tooltip">PA 只给各品种的月度收益贡献（百分点），SGD 金额是估算。利息、费用来自逐笔明细，精确。期货、期货期权、外汇合并为一项（剩余值），因为 PA 与 IBKR 对行权相关盈亏的归类不同（K3）。</div></div></div><div class="cc-body">`;
  if(A.status==='ok'&&!A.value.validated) h+=`<div class="banner">${esc(A.warnings.filter(x=>x.includes('未经验证')).join('；'))}</div>`;
  h+=A.status==='ok'?attrBars(A.value):`<div class="banner">${esc(A.warnings.join('；'))}</div>`;
  return h+`</div></div>`;
}
function drawWin(w){
  const S=D.windows[w].summary; if(S.status!=='ok') return; const v=S.value;
  lineChart(document.getElementById('nav-'+w),[{name:'资产',color:COL[0],pts:v.nav_series}],(n)=>fmt(n/1000)+'k',v.flows);
  const rc=v.return_curves, names=['account',...BM];
  lineChart(document.getElementById('ret-'+w),names.map((k,i)=>({name:k==='account'?'账户':k,color:COL[i],w:k==='account'?2.2:1.4,pts:rc[k]})),(n,dp=0)=>n.toFixed(dp)+'%');
  stackChart(document.getElementById('alc-'+w),v.allocation);
}
function yearly(){
  const rows=D.yearly.value, hd=['年份','期初资产','净转入','盈亏','TWR','股票/ETF','股票期权','期货/期权/外汇','利息','费用'];
  return `<div class="cc"><div class="cc-head"><div class="cc-title">按自然年</div></div><div class="cc-body"><table class="tbl"><tr>${hd.map(h=>`<th>${h}</th>`).join('')}</tr>`+
    rows.map(r=>`<tr><td>${r.year}${r.year===rows[rows.length-1].year?' (YTD)':''}${r.year===rows[0].year?' (8月起)':''}</td><td>${fmt(r.start_nav)}</td><td>${sgn(r.net_flow)}</td><td class="${cls(r.profit)}">${sgn(r.profit)}</td><td class="${cls(r.twr_pct)}">${pc(r.twr_pct,1)}</td>`+
    ['stocks_etf','stock_options','deriv_fx','interest','fees'].map(k=>`<td class="${(!r.validated&&k!=='interest'&&k!=='fees')?'muted':cls(r.buckets[k])}">${sgn(r.buckets[k])}</td>`).join('')+`</tr>`).join('')+
    `</table><div class="fine">归因口径同各窗口：利息和费用精确；灰色数字 = 2025 年以前的估算，未经验证，仅供参考。2017 年从 8 月开户。</div></div></div>`;
}
function notes(){
  const C=D.capabilities;
  return `<div class="cc g1"><div class="cc-head"><div class="cc-title">数据来源与已知约束</div></div><div class="cc-body"><div class="fine">${esc(C.source)} · 报告区间 ${C.period[0]} ~ ${C.period[1]} · 最后一个月是部分月</div>`+
    C.constraints.map(k=>`<div class="k"><b>${k.id} ${esc(k.title)}</b><br>${esc(k.detail)}<br><span class="muted">影响：${esc(k.affects.join('、'))}</span></div>`).join('')+`</div></div>`+
    `<div class="cc g1"><div class="cc-head"><div class="cc-title">导入校验</div></div><div class="cc-body">`+C.import_checks.map(c=>`<div class="k">${c.ok?'✓':'✗'} ${c.id} ${esc(c.desc)}<span class="muted">（${fmt(c.a,3)} / ${fmt(c.b,3)}）</span></div>`).join('')+`</div></div>`+
    `<div class="cc g1"><div class="cc-head"><div class="cc-title">快照历史</div></div><div class="cc-body">`+C.snapshots.map(s=>`<div class="k">#${s.id} 截止 ${s.period_end} · 导入于 ${esc(s.imported_at)} · ${esc(s.file_name)}</div>`).join('')+`</div></div>`;
}
document.getElementById('sub').textContent=`账户 ****${D.account.slice(-4)} · 数据截至 ${D.as_of}（PA 自开户报告）· 基础币种 ${CCY} · 生成于 ${D.generated_at}`;
const failed=D.capabilities.import_checks.filter(c=>!c.ok);
if(failed.length) document.getElementById('banners').innerHTML=`<div class="banner">导入校验未通过：${failed.map(c=>esc(c.id+' '+c.desc)).join('；')}</div>`;
const tabs=document.getElementById('tabs'),panels=document.getElementById('panels');
const ORDER=D.windows_order.map(w=>[w,D.labels[w]]).concat([['YEARLY','分年'],['NOTES','数据说明']]);
ORDER.forEach(([k,label],i)=>{
  const b=document.createElement('button');b.className='tab'+(i==0?' active':'');b.textContent=label;tabs.appendChild(b);
  const p=document.createElement('div');p.className='panel'+(i==0?' active':'');
  p.innerHTML=k==='YEARLY'?yearly():k==='NOTES'?notes():winPanel(k);panels.appendChild(p);
  if(k!=='YEARLY'&&k!=='NOTES') drawWin(k);
  b.onclick=()=>{document.querySelectorAll('.tab,.panel').forEach(x=>x.classList.remove('active'));b.classList.add('active');p.classList.add('active');};
});
</script></body></html>"""


def build_payload(store: Store, rep=None) -> dict:
    snaps = store.snapshots()
    source = store if rep is None else rep
    info = store.get_account_info()
    return {"account": info["account"], "currency": info["base_currency"], "as_of": info["period_end"],
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "benchmarks": info["benchmarks"],
            "windows_order": list(WINDOWS), "labels": WINDOW_LABEL,
            "windows": {w: {"summary": window_summary(source, w).to_dict(), "attribution": window_attribution(source, w).to_dict()}
                        for w in WINDOWS},
            "yearly": yearly_table(source).to_dict(), "capabilities": get_capabilities(source, snaps)}


def main():
    ap = argparse.ArgumentParser(description="IBKR 账户看板（PA 自开户报告）")
    ap.add_argument("--db", default="data/pa.db")
    ap.add_argument("--raw-dir", default="data/raw", help="原始报告归档目录")
    ap.add_argument("--ingest", nargs="*", default=[], help="要导入的 PA 自开户报告")
    ap.add_argument("--force", action="store_true", help="校验未通过也强制导入")
    ap.add_argument("--out", default="dashboard.html")
    a = ap.parse_args()
    Path(a.db).parent.mkdir(parents=True, exist_ok=True)
    store = Store(a.db, a.raw_dir)
    for f in a.ingest:
        r = store.ingest(f, force=a.force)
        print(f"[{r['action']}] {f}" + (f"  {r['reason']}" if r["action"] == "rejected" else ""))
    info = store.get_account_info()
    Path(a.out).write_text(HTML.replace("__DATA__", json.dumps(build_payload(store), ensure_ascii=False)), encoding="utf-8")
    print(f"已生成 {a.out}（数据截至 {info['period_end']}）")


if __name__ == "__main__":
    main()
