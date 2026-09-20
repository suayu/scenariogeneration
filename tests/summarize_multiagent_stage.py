"""汇总冻结场景的四模式结果，并生成中文诊断图与英文论文图。"""
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


MODES=('single','conditional_critic','parallel_arbiter','parallel_memory')
LABELS_ZH={'single':'单 Agent','conditional_critic':'单 Agent＋按需质疑',
           'parallel_arbiter':'双提案＋仲裁','parallel_memory':'仲裁＋离线复盘'}


def llm_metrics(trace_path):
    outputs=[]
    for line in trace_path.open(encoding='utf-8'):
        row=json.loads(line)
        if row.get('kind')=='llm_output':
            outputs.append(row)
    wall=sum(float(row.get('planner_wall_seconds') or 0) for row in outputs)
    api_calls=tokens=0
    for row in outputs:
        trace=row.get('planner_trace') or {}
        events=trace.get('events') or []
        if events:
            for event in events:
                calls=event.get('api_calls') or []
                api_calls+=len(calls)
                for call in calls:
                    tokens+=int((call.get('usage') or {}).get('total_tokens') or 0)
        elif not row.get('request_failed') and row.get('validated_plan') is not None:
            # 旧版单 Agent trace 没有细粒度 api_calls；一次 llm_output 对应一次模型请求。
            api_calls+=1
    return wall,api_calls,tokens


def collect(batches):
    rows=[]
    for scene_index,batch in enumerate(batches):
        batch=Path(batch)
        for result_path in batch.glob('**/attempt_result.json'):
            mode=result_path.parents[2].name
            if mode not in MODES:
                continue
            result=json.loads(result_path.read_text(encoding='utf-8'))
            trace=Path(result['feasibility']['execution_trace_path'])
            wall,calls,tokens=llm_metrics(trace)
            rows.append(dict(scene_index=scene_index,batch=batch.name,mode=mode,
                attack_plans=int(result.get('attack_plan_count') or 0),
                attack_frames=int(result.get('attack_executed_frames') or 0),
                effective_attack=int((result.get('attack_executed_frames') or 0)>0),
                danger_valid=int(bool(result.get('danger_valid'))),
                D=result.get('scenario_danger_score'),planner_seconds=wall,api_calls=calls,
                total_tokens=tokens,simulation_seconds=result['feasibility'].get('wall_seconds'),
                background_collision_frames=result['feasibility'].get('background_collision_frames',0),
                background_static_collision_frames=result['feasibility'].get('background_static_collision_frames',0),
                generation_failure=result.get('generation_failure')))
    return rows


def aggregate(rows):
    output=[]
    for mode in MODES:
        group=[row for row in rows if row['mode']==mode]
        valid=[row['D'] for row in group if row['D'] is not None]
        output.append(dict(mode=mode,scenes=len(group),effective_attacks=sum(r['effective_attack'] for r in group),
            effective_attack_rate=sum(r['effective_attack'] for r in group)/len(group),
            valid_danger=sum(r['danger_valid'] for r in group),mean_D=float(np.mean(valid)) if valid else None,
            mean_planner_seconds=float(np.mean([r['planner_seconds'] for r in group])),
            mean_api_calls=float(np.mean([r['api_calls'] for r in group])),
            mean_tokens=float(np.mean([r['total_tokens'] for r in group])),
            generation_failures=sum(bool(r['generation_failure']) for r in group),
            background_collision_frames=sum(r['background_collision_frames'] for r in group),
            background_static_collision_frames=sum(r['background_static_collision_frames'] for r in group)))
    return output


def plot(summary,out,language):
    if language=='zh':
        font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
        font_manager.fontManager.addfont(font)
        plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
        labels=[LABELS_ZH[row['mode']] for row in summary]
        titles=('有效攻击覆盖率','平均 LLM 规划时间','每场景底层 API 请求数')
        ylabels=('比例','秒','请求数')
    else:
        labels=['Single','Conditional critic','Parallel proposals','Parallel + memory']
        titles=('Effective attack coverage','Mean LLM planning latency','Underlying API calls per scene')
        ylabels=('Rate','Seconds','Calls')
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),constrained_layout=True)
    values=([r['effective_attack_rate'] for r in summary],
            [r['mean_planner_seconds'] for r in summary],
            [r['mean_api_calls'] for r in summary])
    for ax,title,ylabel,data in zip(axes,titles,ylabels,values):
        bars=ax.bar(range(len(labels)),data,color=['#2878b5','#9b59b6','#e69f00','#3a9d5d'])
        ax.set_title(title); ax.set_ylabel(ylabel); ax.set_xticks(range(len(labels)),labels,rotation=18,ha='right')
        ax.grid(axis='y',alpha=.2)
        for bar,value in zip(bars,data):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height(),f'{value:.2f}',ha='center',va='bottom')
    fig.savefig(out,dpi=170)


def main():
    out=Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=True)
    rows=collect(sys.argv[2:])
    if any(sum(r['mode']==mode for r in rows)!=len(sys.argv[2:]) for mode in MODES):
        raise RuntimeError('每个冻结场景必须包含完整四模式结果')
    summary=aggregate(rows)
    (out/'multiagent_stage_results.json').write_text(json.dumps({'rows':rows,'summary':summary},ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'multiagent_stage_results.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    plot(summary,out/'多Agent阶段性对比_中文.png','zh')
    plot(summary,out/'multiagent_stage_comparison_english.png','en')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
