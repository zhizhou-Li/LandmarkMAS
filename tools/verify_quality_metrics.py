# -*- coding: utf-8 -*-
import os
import re
import pandas as pd
import xml.etree.ElementTree as ET
import glob

# ================= 📂 路径配置 =================
# 1. V_raw (OpenCV Baseline) - 分母
RAW_DIR = r"/Agent/outputs/images/Geo/1.29/round_3/baseline_vec_raw"

# 2. Potrace Nodes (Industry Std) - 对比基准
POTRACE_DIR = r"/Agent/outputs/images/Geo/1.29/round_3/base_vec"

# 3. Ours Nodes (Proposed) - 你的方法
OURS_DIR = r"/Agent/outputs/images/Geo/1.29/round_3/vec_3"

# 输出 CSV 路径
OUTPUT_CSV = r"Z:\python_projects\map_entropy\SymbolGeneration\Agent\outputs\images\Geo\1.29\round_3\Table_5_3_Efficiency_Comparison.csv"


# ================= 🛠️ 工具函数 =================

def count_svg_nodes(svg_path):
    """
    解析 SVG 文件并计算路径中的节点（顶点）总数。
    逻辑：统计 path 数据 d="..." 中的坐标对数量。
    """
    if not os.path.exists(svg_path):
        return None

    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()

        # 处理 SVG 命名空间，例如 {http://www.w3.org/2000/svg}path
        # 使用通配符查找所有 path, polygon, polyline
        nodes_count = 0

        # 1. 处理 <path d="...">
        for elem in root.findall(".//{*}path") + root.findall(".//path"):
            d = elem.get('d')
            if d:
                # 移除字母命令，只保留数字和空格/逗号
                # 这种方法统计坐标对的数量，代表几何顶点数
                # 匹配所有数字（包括小数和负数）
                coords = re.findall(r"[-+]?\d*\.\d+|\d+", d)
                # 坐标通常是成对出现的 (x, y)，所以除以 2
                nodes_count += len(coords) // 2

        # 2. 处理 <polygon points="..."> 和 <polyline points="...">
        for elem in root.findall(".//{*}polygon") + root.findall(".//{*}polyline") + \
                    root.findall(".//polygon") + root.findall(".//polyline"):
            points = elem.get('points')
            if points:
                coords = re.findall(r"[-+]?\d*\.\d+|\d+", points)
                nodes_count += len(coords) // 2

        return nodes_count

    except Exception as e:
        print(f"⚠️ 解析错误 {os.path.basename(svg_path)}: {e}")
        return None


def get_clean_name(filename):
    """
    标准化文件名，用于匹配。
    移除 '_baseline' 后缀和扩展名。
    """
    name = os.path.splitext(filename)[0]
    if name.endswith("_baseline"):
        name = name.replace("_baseline", "")
    return name


# ================= 🚀 主程序 =================

def main():
    print("🚀 开始计算矢量化效率对比表 (Table 5-3)...")

    # 获取 Ours 文件夹中的所有文件作为主列表
    # (以此为准，因为这是我们要展示的最终结果)
    ours_files = [f for f in os.listdir(OURS_DIR) if f.lower().endswith('.svg')]

    data_list = []

    for ours_file in ours_files:
        # 1. 确定文件名匹配键
        base_name = get_clean_name(ours_file)

        # 2. 构建三个文件的完整路径
        path_ours = os.path.join(OURS_DIR, ours_file)

        # Potrace 路径 (假设文件名一致)
        path_potrace = os.path.join(POTRACE_DIR, base_name + ".svg")
        # 如果找不到，尝试查找原名.svg (有时候文件名可能略有不同)
        if not os.path.exists(path_potrace):
            path_potrace = os.path.join(POTRACE_DIR, ours_file)

        # Raw 路径 (文件名通常带有 _baseline)
        path_raw = os.path.join(RAW_DIR, base_name + "_baseline.svg")

        # 3. 计算节点数
        n_ours = count_svg_nodes(path_ours)
        n_potrace = count_svg_nodes(path_potrace)
        n_raw = count_svg_nodes(path_raw)

        # 检查数据完整性
        if n_ours is not None and n_potrace is not None and n_raw is not None:
            # 4. 计算提升率 (Improvement vs Potrace)
            # 公式: (Potrace - Ours) / Potrace
            if n_potrace > 0:
                improv = (n_potrace - n_ours) / n_potrace * 100
            else:
                improv = 0

            # 存入列表
            data_list.append({
                "Landmark Type": base_name,  # 这里先用文件名，您可以在Excel里手动改成 'Complex Bridge' 等分类
                "Raw Nodes (V_raw)": n_raw,
                "Potrace Nodes": n_potrace,
                "Ours Nodes": n_ours,
                "Improvement vs. Potrace": improv
            })
            print(f"✅ 处理完成: {base_name}")
        else:
            print(
                f"❌ 跳过 {base_name}: 文件缺失 (Raw: {os.path.exists(path_raw)}, Potrace: {os.path.exists(path_potrace)})")

    # ================= 📊 生成 DataFrame =================
    if not data_list:
        print("❌ 没有生成有效数据，请检查路径文件名是否匹配。")
        return

    df = pd.DataFrame(data_list)

    # 按照 Improvement 排序（可选）
    # df = df.sort_values(by="Improvement vs. Potrace", ascending=False)

    # ================= ➕ 计算 Average 行 =================
    avg_raw = df["Raw Nodes (V_raw)"].mean()
    avg_potrace = df["Potrace Nodes"].mean()
    avg_ours = df["Ours Nodes"].mean()

    # 平均提升率通常基于平均节点数重新计算，或者取提升率的平均值
    # 这里我们取平均提升率
    avg_improv = df["Improvement vs. Potrace"].mean()

    # 创建 Average 行
    avg_row = pd.DataFrame([{
        "Landmark Type": "Average",
        "Raw Nodes (V_raw)": int(avg_raw),
        "Potrace Nodes": int(avg_potrace),
        "Ours Nodes": int(avg_ours),
        "Improvement vs. Potrace": avg_improv
    }])

    # 合并
    df_final = pd.concat([df, avg_row], ignore_index=True)

    # ================= 💾 格式化并保存 =================
    # 格式化百分比
    df_final["Improvement vs. Potrace"] = df_final["Improvement vs. Potrace"].apply(lambda x: f"+{x:.1f}%")

    # 保存 CSV
    df_final.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 60)
    print(f"🎉 成功！表格已生成: {OUTPUT_CSV}")
    print("=" * 60)
    print(df_final.to_string(index=False))


if __name__ == "__main__":
    main()