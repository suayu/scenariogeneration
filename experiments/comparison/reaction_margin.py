"""按计划记录攻击中的响应空间；不把面积直接解释成反应时间或可避碰证明。"""
import math


def curve_compression(baseline, attack):
    """用同一未来时刻的基线归一化，避免把早期面积小误判为攻击最强。"""
    base = {r['time_seconds']:r['area_m2'] for r in baseline}
    curve = []
    for row in attack:
        a0 = base.get(row['time_seconds'])
        ratio = row['area_m2']/a0 if a0 is not None and a0>0 else None
        curve.append({**row, 'baseline_area_m2':a0, 'remaining_area_ratio':ratio,
                      'compression':None if ratio is None else 1-ratio})
    valid = [r for r in curve if r['remaining_area_ratio'] is not None]
    bottleneck = min(valid,key=lambda r:r['remaining_area_ratio']) if valid else None
    return {'curve':curve,'minimum_remaining_area_ratio':None if bottleneck is None else bottleneck['remaining_area_ratio'],
            'bottleneck_future_seconds':None if bottleneck is None else bottleneck['time_seconds']}


def summarize_attacks(records):
    """区分攻击执行时间曲线与每帧预测时域曲线；缺失测量保留null。"""
    groups = {}
    for record in records:
        attack = record.get('attack')
        if not attack:
            continue
        key = (attack.get('source_step'), attack.get('target_id'), attack.get('strategy'))
        group = groups.setdefault(key, {'attack':attack, 'area_over_attack_frames':[],
                                         'forecast_area_curves':[]})
        reach = record.get('reachability')
        # 面积有效性与D有效性分开；EA无效不能抹去已成功计算的面积。
        valid = bool(reach and math.isfinite(reach.get('dangerous_area_m2',float('nan'))))
        source_step=record.get('source_step',None if reach is None else reach['source_step'])
        group['area_over_attack_frames'].append({
            'source_step':source_step,
            'time_since_attack_seconds':None if source_step is None else
                (source_step-(attack.get('source_step') or 0))*(record.get('frame_dt') or .1),
            'D_valid':bool(record.get('valid')),
            'area_m2':reach['dangerous_area_m2'] if valid else None,
            'original_area_m2':reach['original_area_m2'] if valid else None,
            'valid':valid, 'measurement_status':record.get('measurement_status')})
        if reach:
            group['forecast_area_curves'].append({'source_step':reach['source_step'],
                'baseline':reach.get('original_area_curve',[]),
                'attack':reach.get('dangerous_area_curve',[]),
                'minimum_future_area_m2':reach.get('minimum_dangerous_area_m2')})
    for group in groups.values():
        values = [r['area_m2'] for r in group['area_over_attack_frames'] if r['valid']]
        future = [c['minimum_future_area_m2'] for c in group['forecast_area_curves']
                  if c['minimum_future_area_m2'] is not None]
        group['minimum_terminal_area_during_attack_m2'] = min(values) if values else None
        group['minimum_future_area_during_attack_m2'] = min(future) if future else None
        group['valid_frame_count'] = len(values)
        group['observed_frame_count'] = len(group['area_over_attack_frames'])
        group['interpretation'] = 'theoretical_response_space; not_evasion_witness_or_reaction_time'
    return list(groups.values())
