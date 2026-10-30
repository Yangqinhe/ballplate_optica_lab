"""One plotting style for all stages; fixed units/ranges, PNG and vector PDF."""
import html
import json
from pathlib import Path

import numpy as np


COLORS = {"real": "#222222", "baseline": "#D55E00", "fitted": "#0072B2", "prediction": "#009E73",
          "wide_baseline_zero_wavefront": "#D55E00", "wide_fitted_shared_wavefront": "#0072B2",
          "pre_capture_frozen": "#009E73"}


def plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    return plt


def save_figure(fig, base, dpi):
    for suffix in (".png", ".pdf"):
        target = Path(str(base)+suffix)
        if target.exists():
            raise FileExistsError(target)
        fig.savefig(target, dpi=dpi, bbox_inches="tight")


def cell_figure(folder, cell_id, real_stack, models, regions, protocol):
    plt = plotting()
    display = protocol["display"]
    low, high = display["ccd_adu_limits"]
    diff_limit = display["difference_adu_limit"]
    names = list(models)
    real_mean = np.asarray(real_stack, float).mean(0)
    fig, axes = plt.subplots(2, len(names)+1, figsize=(3.0*(len(names)+1), 5.7), squeeze=False)
    im = axes[0, 0].imshow(real_mean, cmap="gray", vmin=low, vmax=high)
    axes[0, 0].set_title("Historical real CCD mean\nN="+str(len(real_stack)))
    axes[1, 0].imshow(np.asarray(real_stack)[0], cmap="gray", vmin=low, vmax=high)
    axes[1, 0].set_title("Historical real single frame #0\nnot averaged control input")
    for column, name in enumerate(names, 1):
        axes[0, column].imshow(models[name], cmap="gray", vmin=low, vmax=high)
        axes[0, column].set_title(name+" model / ADU")
        difference = np.asarray(models[name], float)-real_mean
        dm = axes[1, column].imshow(difference, cmap="RdBu_r", vmin=-diff_limit, vmax=diff_limit)
        axes[1, column].set_title("Model - measured mean / ADU\nfixed scale, no per-image fit")
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    if regions is not None:
        labels = np.asarray(regions).argmax(0)
        # Only overlay the fixed region boundaries, never resize the array.
        for ax in axes[0]:
            ax.contour(labels, levels=np.arange(.5, 7.5), colors="#56B4E9", linewidths=.3, alpha=.55)
    fig.colorbar(im, ax=list(axes[0]), fraction=.015, pad=.015, label="Raw ADU (display only)")
    if names:
        fig.colorbar(dm, ax=list(axes[1, 1:]), fraction=.02, pad=.015, label="ADU")
    fig.suptitle(cell_id+" | HISTORICAL DATA / SEEN MODEL EVALUATION", fontsize=10)
    save_figure(fig, Path(folder)/cell_id, display["dpi"])
    plt.close(fig)


def summary_figures(folder, cells, probes, protocol):
    plt = plotting(); dpi = protocol["display"]["dpi"]
    names = list(dict.fromkeys(name for c in cells for name in c["models"]))
    x = np.arange(len(cells))
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.2), sharex=True)
    fields = [("NCC", "NCC (shape only)"),
              ("raw_RMSE_ADU", "Raw model error / ADU"),
              ("native8_raw_error_over_measured_net_signal_L2", "Raw 8-sum error / net signal")]
    for j, (field, label) in enumerate(fields):
        for name in names:
            # A model role can bind different frozen parents across batches.
            # Never connect those batches into one apparent response curve.
            for bindex, batch in enumerate(dict.fromkeys(c["batch"] for c in cells)):
                selected = [(i,c) for i,c in enumerate(cells) if c["batch"] == batch]
                values = [c["models"].get(name, {}).get(field) for _,c in selected]
                values = [np.nan if v is None else v for v in values]
                axes[j].plot([i for i,_ in selected], values, "o-", ms=3, lw=.9,
                    color=COLORS.get(name), label=name if bindex == 0 else None)
        axes[j].set_ylabel(label); axes[j].grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False, ncol=max(1, len(names)))
    axes[-1].set_xticks(x, [c["id"] for c in cells], rotation=70, ha="right", fontsize=6)
    fig.suptitle("All historical cells | no frame-wise alignment, normalization or filtering")
    fig.tight_layout(); save_figure(fig, Path(folder)/"01_model_gap_all_cells", dpi); plt.close(fig)
    if probes:
        fig, axes = plt.subplots(len(probes), 1, figsize=(8, 2.25*len(probes)), squeeze=False)
        for ax, probe in zip(axes[:, 0], probes):
            metrics = probe["metrics"]
            ax.plot(np.arange(8), metrics["native8_response_ADU"], "o-", color=COLORS["real"], label="Measured mean difference")
            if "model_native8_response_ADU" in metrics:
                ax.plot(np.arange(8), metrics["model_native8_response_ADU"], "s--", color=COLORS["prediction"], label="Frozen model difference")
            ax.axhline(0, color=".7", lw=.5); ax.set_ylabel("Delta region sum / ADU")
            ax.set_title(probe["id"]+" | "+probe.get("kind", "descriptive pair"), fontsize=8)
            ax.legend(frameon=False, fontsize=7); ax.set_xticks(range(8)); ax.set_xlabel("Fixed region index")
        fig.tight_layout(); save_figure(fig, Path(folder)/"02_probe_response_all_pairs", dpi); plt.close(fig)


def number(value, digits=4):
    return "NA" if value is None else format(value, "."+str(digits)+"g")


def write_reports(folder, result):
    root = Path(folder)
    lines = ["# 平台差距与SLM调制：历史数据复算总结", "",
             "本报告是旧实测数据与冻结预测的离线复算，不是新实测，不是独立留出验收。没有连接器件、训练或改变权重。", "",
             "## 当前结论", "", "**无法据此确认平台满足离线部署要求；绝对调制／光学效率尚未测定。**", "",
             "缺少可证明的曝光—图案因果帧关联、新输入／新mask／时间块独立验证，以及功率测量。历史数据可定位误差和探针响应，但不自动放行。", "",
             "历史DARK文件实际是DMD／SLM全0的光学背景，含传感器黑电平与杂散等，并非已验证的物理遮光暗场。下文净信号仅指减去该匹配背景，不声称测得相机本征暗噪声。", "",
             "本批归档的独特信号帧 "+str(result["unique_real_signal_frames"])+" 张、独特全0背景帧 "+str(result["unique_zero_output_background_frames"])+" 张；背景在多个cell复用，不能把归档副本重复计为新样本。", "",
             "## 全部cell，不按结果挑选", "",
             "| Cell | 模型 | NCC | RMSE / ADU | 8区原始误差／实测净信号L2 | 原头动作差MAE / rad |", "|---|---|---:|---:|---:|---:|"]
    for cell in result["cells"]:
        for name, metrics in cell["models"].items():
            lines.append("| "+" | ".join([cell["id"], name, number(metrics["NCC"]), number(metrics["raw_RMSE_ADU"]),
                number(metrics.get("native8_raw_error_over_measured_net_signal_L2")), number(metrics.get("raw_head_mean_action_MAE_rad"))])+" |")
    lines += ["", "![全部cell](figures/01_model_gap_all_cells.png)", "",
              "NCC只刻画形状，不能替代绝对ADU、8区特征与动作误差；原头诊断不是闭环结果。分母接近0时为NA，不补零。8区相对误差以实测全0背景扣除后的净特征作为分母，分子仍是未做逐图拟合的raw误差。背景元数据未匹配的cell不给净信号指标。pre_capture_frozen是各批次模型角色名，不表示跨批次为同一冻结模型。", "",
              "## SLM探针响应", "", "| Pair | 类型 | 实测响应／单帧差分重复噪声RMS | 模型—实测响应余弦 | 响应相对L2误差 |", "|---|---|---:|---:|---:|"]
    for probe in result["probes"]:
        m = probe["metrics"]
        lines.append("| "+" | ".join([probe["id"], probe.get("kind", "pair"), number(m["native8_response_over_single_repeat_RMS"]),
                      number(m.get("model_to_real_response_cosine")), number(m.get("response_relative_L2_error"))])+" |")
    if result["probes"]:
        lines += ["", "![全部探针对照](figures/02_probe_response_all_pairs.png)"]
    lines += ["", "这里的响应比是描述性量，不是统计显著性、绝对衍射效率或可部署证明；分母是sqrt(varA+varB)的单帧参考尺度，不是均值标准误或实测零效应差分噪声。跨cell协方差未测，未自动移除采集间漂移。总ROI亮度不变也不等于SLM未调制——信息可能在区域之间重新分配。", "",
              "## 原始数据与复现", "", "- `raw/`：原始uint8真实单帧与全0背景的独立压缩副本。", "- `derived/`：冻结模型ADU预测与区域，不冒充实测。",
              "- `cells.json`／`probes.json`：全部数值、逐cell来源与缺失证据。", "- `provenance/`：输入配置、原数据SHA与代码版本。",
              "- `MANIFEST.json`：本实验全部文件SHA，不允许覆盖重跑。", "",
              "## 下一步", "", "先预声明并完成可信采集、宏像素／空间频率／灰度响应探针，再在新input×mask×time留出组验证sim-real差距；门槛满足才做直接部署微调。若失败，再校准固定误差与随机／漂移模型，离线重训SLM并重复同一验收，不增加在线电子图像网络。", ""]
    text = "\n".join(lines)
    with (root/"总结.md").open("x", encoding="utf-8") as stream:
        stream.write(text)
    # Self-contained navigation and local figures; no server, JS CDN or device APIs.
    rows = []
    for cell in result["cells"]:
        cid = html.escape(cell["id"])
        metrics = html.escape(json.dumps({"real":cell["real"], "models":cell["models"]}, ensure_ascii=False, indent=2))
        rows.append(f'<details><summary>{cid}</summary><img src="figures/{cid}.png" alt="{cid}"><pre>{metrics}</pre></details>')
    body = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Optical Lab · 历史评估</title>
<style>body{{font:16px system-ui;max-width:1280px;margin:32px auto;padding:0 20px;color:#222}}.warning{{background:#fff3d4;padding:16px;border-left:5px solid #b46a00}}img{{width:100%;height:auto}}details{{border-top:1px solid #ddd;padding:12px}}summary{{cursor:pointer;font-weight:600}}pre{{white-space:pre-wrap;font:12px monospace}}nav a{{margin-right:20px}}label{{display:block;margin:16px 0}}</style>
<h1>平台差距与SLM调制 · 证据查看器</h1><p class="warning">历史真实CCD＋冻结模型预测，不是实时采集。未连接DMD／SLM／CCD；绝对效率未知，实机部署未放行。</p>
<nav><a href="总结.md">总结.md</a><a href="RESULT.json">全部结果</a><a href="MANIFEST.json">数据完整性</a></nav>
<h2>所有cell</h2><img src="figures/01_model_gap_all_cells.png"><p>每组比较使用同一ADU显示范围；不逐图自动对比度或配准。展开查看实际单帧、实测均值、模型、差值与8区。</p>
{''.join(rows)}<h2>探针差分</h2>'''
    if result["probes"]:
        body += '<img src="figures/02_probe_response_all_pairs.png">'
    body += '<p>论文用PNG和矢量PDF在figures/。原始ADU与模型预测分别归档，显示不参与评分。</p></html>'
    with (root/"dashboard.html").open("x", encoding="utf-8") as stream:
        stream.write(body)
