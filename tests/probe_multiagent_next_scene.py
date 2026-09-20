"""按冻结清单顺序检查下一场景，保留上一场景失败而非替换结果。"""
import sys
import launch_multiagent_matrix as matrix

matrix.BASE = matrix.ROOT/'experiments/riskweaver_multiagent_closed_20260917_r2'
matrix.SMOKE_SCENE_INDEX = 4
sys.argv = [__file__, '--phase', 'smoke']
matrix.main()
