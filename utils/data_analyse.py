import json
from typing import Any, List, Union

def print_tree(data: Any, indent: int = 0, key: str = None):
    """
    递归打印嵌套结构的树状视图（仅显示结构和形状，不输出具体内容）。
    """
    prefix = "  " * indent
    if key is not None:
        prefix += f"{key}: "

    if isinstance(data, dict):
        print(f"{prefix}dict (size={len(data)})")
        for k, v in data.items():
            print_tree(v, indent + 1, k)
    elif isinstance(data, list):
        print(f"{prefix}list (length={len(data)})")
        for idx, item in enumerate(data):
            print_tree(item, indent + 1, f"[{idx}]")
    elif isinstance(data, str):
        print(f"{prefix}str (len={len(data)})")          # 只显示长度
    elif isinstance(data, int):
        print(f"{prefix}int")
    elif isinstance(data, float):
        print(f"{prefix}float")
    elif isinstance(data, bool):
        print(f"{prefix}bool")
    elif data is None:
        print(f"{prefix}None")
    else:
        # 兜底其他类型
        print(f"{prefix}{type(data).__name__}")

def find_roadedge(data: Any, results: List[Any]):
    """
    递归查找所有键（忽略大小写）为 'roadedge' 的值，存入 results。
    """
    if isinstance(data, dict):
        for k, v in data.items():
            if k.lower() == "roadedge":
                results.append(v)
            # 继续递归遍历值（因为值可能是嵌套结构）
            find_roadedge(v, results)
    elif isinstance(data, list):
        for item in data:
            find_roadedge(item, results)
    # 基本类型无子节点，忽略

def analyse_file(file_path: str):
    """
    从 JSON 文件读取数据，执行树状打印和搜索 roadEdge。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("=" * 50)
    print("树状结构解析：")
    print_tree(data)

    print("\n" + "=" * 50)
    print("搜索键为 'roadEdge'（不区分大小写）的数据：")
    roadedge_values = []
    find_roadedge(data, roadedge_values)
    if roadedge_values:
        for idx, val in enumerate(roadedge_values, 1):
            print(f"找到 #{idx}: {repr(val)}")
    else:
        print("未找到任何匹配的键。")

def analyse_json(data: Any):
    """
    从 JSON 文件读取数据，执行树状打印和搜索 roadEdge。
    """

    print("=" * 50)
    print("树状结构解析：")
    print_tree(data)

    print("\n" + "=" * 50)
    print("搜索键为 'roadEdge'（不区分大小写）的数据：")
    roadedge_values = []
    find_roadedge(data, roadedge_values)
    if roadedge_values:
        for idx, val in enumerate(roadedge_values, 1):
            print(f"找到 #{idx}: {repr(val)}")
    else:
        print("未找到任何匹配的键。")

if __name__ == "__main__":
    # 请替换为你的文件路径
    analyze_file("your_data.json")