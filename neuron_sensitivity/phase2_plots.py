"""Publication-oriented SVG figures; SVG keeps plotting dependency-free."""
from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd


COLORS = {"fc1": "#3569a8", "fc2": "#dd7435", "incoming_norm_only": "#6a51a3",
          "all_pre_ablation_properties": "#238b75", "seed": "#5078a5"}
W, H = 1100, 700


def _text(x, y, value, size=14, anchor="middle", weight="normal", rotate=None):
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}"{transform}>{escape(str(value))}</text>'


def _start(title: str) -> list[str]:
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#202833}.grid{stroke:#dce2e9;stroke-width:1}.axis{stroke:#34404d;stroke-width:1.5}</style>',
            _text(W/2, 38, title, 23, weight="bold"),
            _text(W/2, 62, "MNIST test set · 10,000 examples/model · 5 independent seeds · fc1: 128 units, fc2: 64 units", 12)]


def _finish(path: Path, svg: list[str]) -> None:
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def _axis(svg: list[str], x0: float, y0: float, pw: float, ph: float,
          xmin: float, xmax: float, ymin: float, ymax: float,
          xlabel: str, ylabel: str, xfmt=".2f", yfmt=".2f") -> None:
    for i in range(6):
        t=i/5; x=x0+pw*t; y=y0+ph-ph*t
        xv=xmin+(xmax-xmin)*t; yv=ymin+(ymax-ymin)*t
        svg.extend([f'<line class="grid" x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0+ph}"/>',
                    f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x0+pw}" y2="{y:.1f}"/>',
                    _text(x, y0+ph+23, format(xv, xfmt), 12),
                    _text(x0-12, y+4, format(yv, yfmt), 12, anchor="end")])
    svg.extend([f'<line class="axis" x1="{x0}" y1="{y0+ph}" x2="{x0+pw}" y2="{y0+ph}"/>',
                f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+ph}"/>',
                _text(x0+pw/2, y0+ph+55, xlabel, 15),
                _text(28, y0+ph/2, ylabel, 15, rotate=-90)])


def _scatter_svg(path: Path, frame: pd.DataFrame, xcol: str, ycol: str,
                 title: str, xlabel: str, ylabel: str,
                 group_col="layer", group_colors=None, xlim=None, ylim=None) -> None:
    svg=_start(title); x0,y0,pw,ph=120,85,790,500
    xmin,xmax=xlim or (float(frame[xcol].min()),float(frame[xcol].max()))
    ymin,ymax=ylim or (float(frame[ycol].min()),float(frame[ycol].max()))
    if xmax<=xmin: xmax=xmin+1
    if ymax<=ymin: ymax=ymin+1
    _axis(svg,x0,y0,pw,ph,xmin,xmax,ymin,ymax,xlabel,ylabel,".3f",".3f")
    colors=group_colors or COLORS
    groups=frame.groupby(group_col) if group_col in frame else [("all",frame)]
    for gi,(name,rows) in enumerate(groups):
        color=colors.get(name,"#557b6e")
        for row in rows.itertuples():
            xv=getattr(row,xcol); yv=getattr(row,ycol)
            x=x0+pw*(xv-xmin)/(xmax-xmin); y=y0+ph-ph*(yv-ymin)/(ymax-ymin)
            svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.6" fill="{color}" fill-opacity=".58"/>')
        ly=110+gi*26
        svg.extend([f'<circle cx="970" cy="{ly}" r="5" fill="{color}"/>',_text(985,ly+5,name,13,anchor="start")])
    _finish(path,svg)


def make_phase2_plots(frame: pd.DataFrame, analyses: dict[str,pd.DataFrame], out_dir: str|Path) -> list[Path]:
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); made=[]
    # 1) Ten rows: one sensitivity heatmap for each layer/model pair.
    f=frame; svg=_start("MNIST neuron ablation sensitivity across five training seeds")
    left,top,cellw,rowh=220,95,5.3,34
    vmax=max(float(f.prediction_flip_pct.max()),1e-12)
    def heat(v):
        t=min(1.,v/vmax); a=(247,244,239); b=(130,25,45)
        return "#%02x%02x%02x"%tuple(round(a[i]*(1-t)+b[i]*t) for i in range(3))
    rows=0
    for seed in sorted(f.seed.unique()):
        for layer in ("fc1","fc2"):
            grp=f[(f.seed==seed)&(f.layer==layer)].sort_values("neuron")
            y=top+rows*rowh; svg.append(_text(left-14,y+22,f"seed {seed}  {layer}",13,anchor="end"))
            for r in grp.itertuples():
                svg.append(f'<rect x="{left+r.neuron*cellw:.1f}" y="{y}" width="{cellw-0.5:.1f}" height="25" fill="{heat(100*r.prediction_flip_rate)}"/>')
            rows+=1
    svg.extend([_text(left,top+rows*rowh+12,"Neuron index: fc1 has 128 units; fc2 has 64 units",13,anchor="start"),
                _text(left,top+rows*rowh+44,f"Flip rate scale: 0–{vmax:.2f}%",13,anchor="start")])
    p=out/"phase2_sensitivity_heatmaps.svg";_finish(p,svg);made.append(p)

    # 2–5) Weight/activation properties against functional sensitivity.
    for col,title,xlabel,name in [
        ("incoming_weight_norm","Incoming weight norm vs sensitivity","Incoming weight L2 norm","incoming_norm_vs_sensitivity.svg"),
        ("outgoing_weight_norm","Outgoing weight norm vs sensitivity","Outgoing weight L2 norm","outgoing_norm_vs_sensitivity.svg"),
        ("activation_frequency","Activation frequency vs sensitivity","Fraction of test examples with positive activation","activation_frequency_vs_sensitivity.svg"),
        ("mean_activation","Mean activation vs sensitivity","Mean ReLU activation over test examples","mean_activation_vs_sensitivity.svg"),
    ]:
        p=out/name; _scatter_svg(p,frame,col,"prediction_flip_rate",title,xlabel,"Ablation prediction-flip rate (fraction)",ylim=(0,max(float(frame.prediction_flip_rate.max())*1.05,1e-4)));made.append(p)

    # 6) Correlations by feature, layer; dots and error bars summarize five models.
    corr=analyses["correlations"]; features=list(corr.feature.drop_duplicates())
    svg=_start("Feature-wise Spearman correlation with ablation sensitivity")
    x0,y0,pw,ph=150,95,780,420; ymin,ymax=-1,1
    for i in range(5):
        t=i/4; y=y0+ph-ph*t; val=ymin+(ymax-ymin)*t
        svg.extend([f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x0+pw}" y2="{y:.1f}"/>',_text(x0-12,y+4,f"{val:.1f}",12,anchor="end")])
    svg.extend([f'<line class="axis" x1="{x0}" y1="{y0+ph}" x2="{x0+pw}" y2="{y0+ph}"/>',
                f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+ph}"/>',
                _text(35,y0+ph/2,"Spearman ρ across models",14,rotate=-90)])
    summary=analyses["correlation_summary"]
    feature_labels={"incoming_weight_norm":"incoming norm","outgoing_weight_norm":"outgoing norm",
                    "activation_frequency":"active frequency","mean_activation":"mean activation",
                    "mean_abs_local_logit_contribution":"local logit effect"}
    for i,feature in enumerate(features):
        x=x0+pw*(i+.5)/len(features)
        svg.append(_text(x,y0+ph+24,feature_labels.get(feature,feature),12))
        for j,layer in enumerate(("fc1","fc2")):
            row=summary[(summary.feature==feature)&(summary.layer==layer)].iloc[0]
            rho=float(row.mean_rho); sd=float(row.sd_across_models) if pd.notna(row.sd_across_models) else 0
            xx=x+(j-.5)*13; yy=y0+ph-ph*(rho-ymin)/(ymax-ymin)
            ylo=y0+ph-ph*(max(ymin,rho-sd)-ymin)/(ymax-ymin);yhi=y0+ph-ph*(min(ymax,rho+sd)-ymin)/(ymax-ymin)
            svg.extend([f'<line x1="{xx}" y1="{ylo}" x2="{xx}" y2="{yhi}" stroke="{COLORS[layer]}" stroke-width="2"/>',
                        f'<circle cx="{xx}" cy="{yy}" r="6" fill="{COLORS[layer]}"/>'])
    svg.extend([_text(x0,y0+ph+62,"Neuron properties",14,anchor="start"),
                f'<circle cx="740" cy="615" r="5" fill="{COLORS["fc1"]}"/>',_text(755,620,"fc1 mean ± SD across 5 models",12,anchor="start"),
                f'<circle cx="950" cy="615" r="5" fill="{COLORS["fc2"]}"/>',_text(965,620,"fc2 mean ± SD across 5 models",12,anchor="start")])
    p=out/"feature_spearman_comparison.svg";_finish(p,svg);made.append(p)

    # 7) Out-of-model predictions, each point generated by a fold that held out its seed.
    pred=analyses["predictions"]; svg=_start("Held-out-model prediction of neuron sensitivity")
    lo=0.;hi=max(float(pred.observed_flip_rate.max()),float(pred.predicted_flip_rate.max()),1e-4)*1.05
    x0,y0,pw,ph=120,85,790,500;_axis(svg,x0,y0,pw,ph,lo,hi,lo,hi,"Observed flip rate (fraction)","Predicted flip rate (fraction)",".3f",".3f")
    svg.append(f'<line x1="{x0}" y1="{y0+ph}" x2="{x0+pw}" y2="{y0}" stroke="#555" stroke-dasharray="7,5"/>')
    for name,grp in pred.groupby("model"):
        for r in grp.itertuples():
            x=x0+pw*r.observed_flip_rate/hi;y=y0+ph-ph*r.predicted_flip_rate/hi
            svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{COLORS[name]}" fill-opacity=".45"/>')
    svg.extend([f'<circle cx="970" cy="110" r="5" fill="{COLORS["incoming_norm_only"]}"/>',_text(985,115,"incoming norm only",12,anchor="start"),
                f'<circle cx="970" cy="136" r="5" fill="{COLORS["all_pre_ablation_properties"]}"/>',_text(985,141,"all pre-ablation properties",12,anchor="start")])
    p=out/"heldout_model_predicted_vs_observed.svg";_finish(p,svg);made.append(p)

    # 8) Box summaries per seed/layer; seed, not neuron, is the replication unit.
    svg=_start("Ablation sensitivity distributions by layer and training seed")
    x0,y0,pw,ph=115,90,850,490; maxy=max(float(frame.prediction_flip_rate.max())*1.05,1e-4)
    # Categorical x positions are labeled per seed/layer, not as numeric values.
    for i in range(5):
        t=i/4; y=y0+ph-ph*t; val=maxy*t
        svg.extend([f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x0+pw}" y2="{y:.1f}"/>',_text(x0-12,y+4,f"{val:.3f}",11,anchor="end")])
    svg.extend([f'<line class="axis" x1="{x0}" y1="{y0+ph}" x2="{x0+pw}" y2="{y0+ph}"/>',
                f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+ph}"/>',
                _text(32,y0+ph/2,"Prediction-flip rate (fraction)",13,rotate=-90)])
    index=0
    for seed in sorted(frame.seed.unique()):
        for layer in ("fc1","fc2"):
            s=frame[(frame.seed==seed)&(frame.layer==layer)].prediction_flip_rate.to_numpy()
            q=np.quantile(s,[0,.25,.5,.75,1]);cx=x0+pw*(index+.5)/10;y=lambda v:y0+ph-ph*v/maxy
            color=COLORS[layer]
            svg += [f'<line x1="{cx}" y1="{y(q[0])}" x2="{cx}" y2="{y(q[4])}" stroke="{color}" stroke-width="2"/>',
                    f'<rect x="{cx-22}" y="{y(q[3])}" width="44" height="{max(1,y(q[1])-y(q[3]))}" fill="{color}" fill-opacity=".28" stroke="{color}"/>',
                    f'<line x1="{cx-22}" y1="{y(q[2])}" x2="{cx+22}" y2="{y(q[2])}" stroke="{color}" stroke-width="3"/>',
                    _text(cx,y0+ph+22,f"{seed} {layer}",11)]
            index+=1
    svg.append(_text(x0+pw/2,y0+ph+63,"Training seed and layer",14))
    p=out/"layerwise_distributions_by_seed.svg";_finish(p,svg);made.append(p)

    # 9) Margin changes for low/high-sensitivity neurons in every seed/layer.
    svg=_start("True-label margin change for selected low- and high-sensitivity neurons")
    selected=[]
    for (seed,layer),g in frame.groupby(["seed","layer"]):
        ordered=g.sort_values("prediction_flip_rate")
        selected.extend([(seed,layer,"low",r) for r in ordered.head(1).itertuples()])
        selected.extend([(seed,layer,"high",r) for r in ordered.tail(1).itertuples()])
    vals=[r.mean_true_label_margin_change for _,_,_,r in selected]
    ymin=min(0,min(vals));ymax=max(0,max(vals));ymax=max(ymax,1e-6);ymin=min(ymin,-1e-6)
    x0,y0,pw,ph=95,100,900,440
    for i in range(5):
        t=i/4; y=y0+ph-ph*t; val=ymin+(ymax-ymin)*t
        svg.extend([f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x0+pw}" y2="{y:.1f}"/>',_text(x0-12,y+4,f"{val:.2f}",11,anchor="end")])
    svg.extend([f'<line class="axis" x1="{x0}" y1="{y0+ph}" x2="{x0+pw}" y2="{y0+ph}"/>',
                f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+ph}"/>',
                _text(26,y0+ph/2,"Change in true-label margin (logit units)",12,rotate=-90)])
    zero=y0+ph-ph*(0-ymin)/(ymax-ymin)
    svg.append(f'<line x1="{x0}" y1="{zero}" x2="{x0+pw}" y2="{zero}" stroke="#333" stroke-width="2"/>')
    slot=pw/len(selected)
    for i,(seed,layer,kind,r) in enumerate(selected):
        x=x0+(i+.5)*slot; yy=y0+ph-ph*(r.mean_true_label_margin_change-ymin)/(ymax-ymin)
        ytop=min(zero,yy); hh=max(1,abs(yy-zero)); color=COLORS[layer] if kind=="high" else "#9aa5b1"
        svg.append(f'<rect x="{x-slot*.32}" y="{ytop}" width="{slot*.64}" height="{hh}" fill="{color}"/>')
        svg.append(_text(x,y0+ph+26,f"{seed} {layer} {kind}",9,rotate=-45))
    svg.append(_text(x0+pw/2,y0+ph+86,"Seed / layer / lowest or highest sensitivity neuron",13))
    p=out/"selected_neuron_margin_changes.svg";_finish(p,svg);made.append(p)
    return made
