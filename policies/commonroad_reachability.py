"""CommonRoad-Reach隔离进程适配器；原网格后端仍是默认实现。"""
import json
import select
import subprocess
import time
from pathlib import Path

from policies.risk_metrics import RouteReachability, _config_value


class CommonRoadReachability(RouteReachability):
    """在独立Python环境传播可达集合，主环境只交换无凭据JSON。"""

    def __init__(self, config=None):
        super().__init__(config)
        self.python = str(_config_value(config, 'commonroad_python',
            '/home2/zhaoyx/venvs/riskweaver-commonroad-reach-2025.2.1/bin/python'))
        self.timeout = float(_config_value(config, 'commonroad_timeout_seconds', 120))
        self.process = None
        self.sequence = 0

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process.stdin.close()
            self.process.stdout.close()
            self.process = None

    def compute(self, env, use_joint_trajectory, static_obstacles=None):
        """同一源帧与完整联合未来；服务错误显式失败，不偷偷回退网格。"""
        if use_joint_trajectory:
            from experiments.comparison.ctrl_sim_forecast import validate_forecast
            validate_forecast(env.pending_joint_trajectory, env, self.horizon)
        if self.process is None:
            worker = Path(__file__).resolve().parents[1] / 'experiments/comparison/commonroad_worker.py'
            self.process = subprocess.Popen([self.python, '-u', str(worker)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.sequence += 1
        steps = int(round(self.horizon / self.dt))
        payload = {'request_id': self.sequence, 'route': env.scenario_dict['route'].tolist(),
            'ego': env.ego_state.tolist(), 'dt': self.dt, 'steps': steps,
            'lane_half_width': self.lane_half_width, 'max_speed': self.max_speed,
            'max_lateral_speed': self.max_lateral_speed,
            'a_lon': [min(self.longitudinal_accelerations), max(self.longitudinal_accelerations)],
            'a_lat': [min(self.lateral_accelerations), max(self.lateral_accelerations)],
            'grid_size': self.s_resolution,
            'dynamic': [[(c.tolist(), y, length, width) for c,y,length,width in
                         self._dynamic_obstacles(env, k*self.dt, use_joint_trajectory)]
                        for k in range(steps+1)],
            'static': list(static_obstacles or [])}
        start = time.monotonic()
        try:
            self.process.stdin.write(json.dumps(payload)+'\n')
            self.process.stdin.flush()
            if not select.select([self.process.stdout], [], [], self.timeout)[0]:
                raise TimeoutError('commonroad_reach_timeout')
            response = json.loads(self.process.stdout.readline())
            if response.get('request_id') != self.sequence:
                raise RuntimeError('commonroad_request_id_mismatch')
            if response.get('error'):
                raise RuntimeError('commonroad_reach_failure: '+response['error'])
            result = response['result']
            result['wall_seconds'] = time.monotonic()-start
            return result
        except Exception:
            self.close()
            raise
