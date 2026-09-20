"""只读预筛原始场景，使用正式画像规则检查初始可行候选。"""
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np

from policies.ego_profile import CandidateBuilder


class EmptyKnowledge:
    def retrieve(self, condition, family):
        return dict(attempts=0, measured_executions=0, failure_count=0,
                    vulnerability_estimate=.5, uncertainty_interval=[0, 1])


def main():
    root = Path('/home2/zhaoyx/scenario-dreamer')
    files = sorted((root/'metadata/simulation_environment_datasets/scenario_dreamer_waymo_200m_pickles').glob('*.pkl'))
    builder = CandidateBuilder(SimpleNamespace(low_speed_decelerations_mps2=[.25,.5]))
    rows = []
    for file in files:
        with file.open('rb') as handle:
            data = pickle.load(handle)
        agents = np.asarray(data['agents'][:-1, 0], float)
        types = np.asarray(data['agent_types'][:-1])
        vehicle = types[:, 1] == 1
        agents = agents[vehicle]
        env = SimpleNamespace(ego_state=np.asarray(data['agents'][-1, 0], float),
                              data_dict={'agent': [agents]}, agent_active=agents[:, 7] > .5,
                              scenario_dict={'route': data['route'], 'lanes': data['lanes']},
                              get_static_obstacles=lambda: [])
        try:
            candidates, rejected = builder.build(env, {}, EmptyKnowledge())
            rows.append(dict(scene=file.name, candidates=len(candidates),
                             target_ids=[c['target_id'] for c in candidates], rejected=rejected,
                             active=int(env.agent_active.sum())))
        except Exception as error:
            rows.append(dict(scene=file.name, error=type(error).__name__, active=int(env.agent_active.sum())))
    print(json.dumps(rows, ensure_ascii=False))


if __name__ == '__main__':
    main()
