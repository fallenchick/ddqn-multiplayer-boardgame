# -*- coding: utf-8 -*-
"""生成 docs/convergence.svg —— 训练收敛曲线（纯标准库，无 matplotlib 依赖）。

数据来自 results/train_ma.csv（100 局滑动平均的最终分数）。
DDQN 侧从 −221 升到 −2（越接近 0 越好），GitHub 侧恒定在 +105 作为对照组。
"""
import csv, os, math

HERE = os.path.dirname(os.path.abspath(__file__))
MA = os.path.join(HERE, 'results', 'train_ma.csv')
OUT = os.path.join(HERE, 'docs', 'convergence.svg')

W, H = 900, 460
ML, MR, MT, MB = 70, 24, 44, 56          # 边距
PW, PH = W - ML - MR, H - MT - MB

def load():
    rows = []
    with open(MA, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            rows.append((float(r['episode']), float(r['ddqn_ma']), float(r['github_ma'])))
    return rows

def main():
    rows = load()
    xs = [r[0] for r in rows]
    dds = [r[1] for r in rows]
    ghs = [r[2] for r in rows]
    x0, x1 = min(xs), max(xs)
    ymin = min(min(dds), min(ghs))
    ymax = max(max(dds), max(ghs))
    pad = (ymax - ymin) * 0.12
    ymin -= pad; ymax += pad

    def px(x): return ML + (x - x0) / (x1 - x0) * PW
    def py(y): return MT + PH - (y - ymin) / (ymax - ymin) * PH

    def path(ys):
        return 'M ' + ' L '.join('%.1f,%.1f' % (px(x), py(y)) for x, y in zip(xs, ys))

    # y 轴刻度
    ticks = []
    for i in range(6):
        v = ymin + (ymax - ymin) * i / 5
        ticks.append((py(v), v))
    # x 轴刻度
    xticks = []
    for i in range(6):
        v = x0 + (x1 - x0) * i / 5
        xticks.append((px(v), v))

    s = []
    s.append('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" font-family="Helvetica,Arial,sans-serif">' % (W, H, W, H))
    s.append('<rect width="%d" height="%d" fill="#ffffff"/>' % (W, H))
    # 网格
    for y, _ in ticks:
        s.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#e8eaed" stroke-width="1"/>' % (ML, y, ML + PW, y))
    # 零线（更重要：越接近 0 越好）
    s.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#9aa0a6" stroke-width="1.5" stroke-dasharray="5,4"/>' % (ML, py(0), ML + PW, py(0)))
    # 轴
    s.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#5f6368" stroke-width="1.5"/>' % (ML, MT + PH, ML + PW, MT + PH))
    s.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#5f6368" stroke-width="1.5"/>' % (ML, MT, ML, MT + PH))
    # 曲线：GitHub 对照（灰，虚线）+ DDQN（蓝）
    s.append('<path d="%s" fill="none" stroke="#9aa0a6" stroke-width="2" stroke-dasharray="6,4"/>' % path(ghs))
    s.append('<path d="%s" fill="none" stroke="#1a73e8" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>' % path(dds))
    # 刻度文字
    for y, v in ticks:
        s.append('<text x="%.1f" y="%.1f" font-size="12" fill="#5f6368" text-anchor="end">%d</text>' % (ML - 8, y + 4, round(v)))
    for x, v in xticks:
        s.append('<text x="%.1f" y="%.1f" font-size="12" fill="#5f6368" text-anchor="middle">%d</text>' % (x, MT + PH + 20, round(v)))
    # 轴标题
    s.append('<text x="%.1f" y="%.1f" font-size="13" fill="#3c4043" text-anchor="middle">训练局数（episode）</text>' % (ML + PW / 2, H - 14))
    s.append('<text x="18" y="%.1f" font-size="13" fill="#3c4043" text-anchor="middle" transform="rotate(-90 18 %.1f)">最终得分（100 局滑动平均）</text>' % (MT + PH / 2, MT + PH / 2))
    # 图例
    lx, ly = ML + 16, MT + 14
    s.append('<rect x="%.1f" y="%.1f" width="210" height="56" fill="#ffffff" stroke="#dadce0" rx="4"/>' % (lx, ly))
    s.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#1a73e8" stroke-width="2.5"/>' % (lx + 12, ly + 17, lx + 42, ly + 17))
    s.append('<text x="%.1f" y="%.1f" font-size="12" fill="#3c4043">DDQN 智能体</text>' % (lx + 50, ly + 21))
    s.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#9aa0a6" stroke-width="2" stroke-dasharray="6,4"/>' % (lx + 12, ly + 39, lx + 42, ly + 39))
    s.append('<text x="%.1f" y="%.1f" font-size="12" fill="#3c4043">GithubAI（对照组，恒定）</text>' % (lx + 50, ly + 43))
    # 标题
    s.append('<text x="%.1f" y="24" font-size="15" font-weight="bold" fill="#202124" text-anchor="middle">训练收敛曲线 · 1,766 局实测（美国地图 2 人局）</text>' % (W / 2))
    s.append('</svg>')

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, 'w', encoding='utf-8').write('\n'.join(s))
    print('✅ %s (%.1f KB) | %d 个数据点' % (OUT, os.path.getsize(OUT) / 1024, len(rows)))

if __name__ == '__main__':
    main()
