"""通过 SSH 标准输入输出领取/回传请求，不开放网络监听端口。"""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policies.codex_planner_bridge import CodexQueueClient, write_json_exclusive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['take', 'deliver'])
    args = parser.parse_args()
    directory = CodexQueueClient().directory
    if args.action == 'take':
        # 原子领取；已领取请求不重跑，避免重复调用或重复消费。
        for pending in sorted(directory.glob('*/pending.json')):
            try:
                pending.rename(pending.with_name('claimed.json'))
            except FileNotFoundError:
                continue
            request = json.loads(pending.with_name('claimed.json').read_text())
            if request['deadline'] <= time.time():
                continue
            print(json.dumps(request, ensure_ascii=False))
            return
        print('null')
    else:
        response = json.load(sys.stdin)
        rid = response.get('request_id', '')
        if not re.fullmatch(r'[a-f0-9]{32}', rid):
            raise ValueError('invalid request_id')
        folder = directory / rid
        request = json.loads((folder / 'claimed.json').read_text())
        if request['request_id'] != rid or request['deadline'] <= time.time():
            raise ValueError('expired or mismatched request')
        write_json_exclusive(folder / 'response.tmp', response)
        # 独占临时文件也充当已投递标志，重复投递不能覆盖结果。
        os.link(folder / 'response.tmp', folder / 'response.json')
        print('delivered')


if __name__ == '__main__':
    main()
