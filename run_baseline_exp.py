# -*- coding: utf-8 -*-
import sys
import os
import base64
import json
import time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import requests

# ==========================================
# 🔧 路径修复补丁
# ==========================================
current_file_path = os.path.abspath(__file__)
agent_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(agent_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ==========================================

# 导入必要模块
try:
    # 我们依然需要 Grounder 来获取“正确答案”给裁判看
    from Agent.agents.grounder_agent import ground_entity_to_spec
    # 我们依然需要 Reviewer 来打分
    from Agent.agents.reviewer_agent import run_reviewer
    from Agent.config import MODELS, OPENAI_API_KEY
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    sys.exit(1)

# ==========================================
# 📂 路径配置
# ==========================================
# 注意：保存到不同的文件夹，方便区分
OUTPUT_BASE_DIR = r"Z:\python_projects\map_entropy\SymbolGeneration\Agent\outputs\images\baseline"
CSV_SAVE_PATH = r"Z:\python_projects\map_entropy\SymbolGeneration\Agent\outputs\images\baseline\experiment_baseline.csv"
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)


# ==========================================

def generate_image_baseline(prompt, save_path):
    """
    [Baseline 黑白版] 强制生成黑白二色调符号
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    model_name = "gpt-image-1-mini"

    # --- 关键修改：Baseline 的黑白风格控制 ---
    # Baseline 输入通常只有地标名，所以这里的后缀决定了画面的全部风格
    base_prompt = (
        f"{prompt}, icon style, black and white line art, "
        "strong black outlines, white background, no complex details, "
        "no filling, monochrome vector graphic."
    )

    print(f"   🎨 Baseline Drawing: {base_prompt[:40]}...")

    try:
        response = client.images.generate(
            model=model_name,
            prompt=base_prompt,
            size="1024x1024",
            n=1
        )

        if hasattr(response.data[0], 'b64_json') and response.data[0].b64_json:
            image_data = base64.b64decode(response.data[0].b64_json)
            with open(save_path, 'wb') as f:
                f.write(image_data)
            return True
        elif hasattr(response.data[0], 'url') and response.data[0].url:
            img_data = requests.get(response.data[0].url, timeout=30).content
            with open(save_path, 'wb') as handler:
                handler.write(img_data)
            return True
        return False
    except Exception as e:
        # Baseline 的安全重试
        if "safety" in str(e).lower():
            print("   ⚠️ Safety triggered, retrying with abstract lines...")
            safe_prompt = f"Abstract black lines forming a shape of {prompt}, white background, geometric symbol."
            try:
                response = client.images.generate(model=model_name, prompt=safe_prompt, size="1024x1024", n=1)
                if hasattr(response.data[0], 'b64_json') and response.data[0].b64_json:
                    image_data = base64.b64decode(response.data[0].b64_json)
                    with open(save_path, 'wb') as f: f.write(image_data)
                    return True
            except:
                pass
        print(f"   ❌ Generation Error: {e}")
        return False


# --- 使用同样的 9 个高难度地标 ---
LANDMARKS = [
    # ========================================================
    # Type I: Complex Bridges (复杂桥梁结构)
    # 测试点：拓扑连通性、非对称结构、缠绕与镂空
    # ========================================================
    # 1. [基准] 非对称拱桥 (Original)
    # "Gateshead Millennium Bridge 盖茨黑德千禧桥",
    # 2. DNA双螺旋 (极难，测试密集镂空)
    # "The Helix Bridge Singapore 新加坡双螺旋桥",
    # 3. 莫比乌斯环 (测试多层垂直交织)
    # "Lucky Knot Bridge Changsha 长沙幸运结桥",
    # 4. 蛇形流线 (测试有机形态)
    #"Python Bridge Amsterdam 阿姆斯特丹巨蟒桥",
    # 5. 波浪形拱 (测试扎哈风格的非对称流动感)
    #  "Sheikh Zayed Bridge Abu Dhabi 谢赫扎耶德大桥",
    # 6. 木质波浪纹理 (测试材质与几何的结合)
    # "Henderson Waves Bridge Singapore 亨德森波浪桥",
    # 7. 垂直起伏如意形 (测试垂直方向的波浪，区别于平面圆环)
    # "Ruyi Bridge Shenxianju 神仙居如意桥",

    # ========================================================
    # Type II: Regular Architecture (规整建筑)
    # 测试点：透视畸变、几何组合、重复纹理、反直觉姿态
    # ========================================================
    # 1. [基准] 尖顶教堂 (Original - 强垂直线条)
    # "Wujin Lotus Conference Center 常州武进莲花馆"
    # 15. 蜂窝楼梯 (测试高频重复纹理与透视)
    # "Wuxi Taihu Show Theatre 无锡太湖秀剧场",

    # "Guangzhou Circle Mansion 广州圆大厦"

    # "Shenyang Fangyuan Mansion 沈阳方圆大厦",
    # 3. 飞碟形状 (测试单点支撑平衡感)
    #"Niterói Contemporary Art Museum 尼泰罗伊当代艺术博物馆",
    # "Fuzhou Strait Culture and Art Centre 福州海峡文化艺术中心"
    # 17. 球体+管道 (测试几何体连接关系)
    # "Atomium Brussels 布鲁塞尔原子塔",
    # 18. 网格+球体 (测试方正与圆形的强行嵌入)
    # "CCTV building 中央电视台大楼",
    # 19. 倒立建筑 (测试反直觉的姿态控制)
     #"WonderWorks Orlando 奥兰多颠倒屋",
    # 20. 倒金字塔/树形 (补位：测试头重脚轻的悬臂结构，区别于翠城)
    # "Geisel Library UCSD 盖泽尔图书馆",

    # ========================================================
    # Type III: Free-form Sculptures (自由形态雕塑)
    # 测试点：语义纠错、生物形态、材质抽象、局部与整体
    # ========================================================
    # 21. [基准] 局部马头 (Original - 测试生物特征放大)
    #  "Maman sculpture Louise Bourgeois 毕尔巴鄂大蜘蛛雕塑",
    # 25. 切片人头 (测试旋转切片的对齐与错位)
    #  "Metalmorphosis Prague 布拉格卡夫卡旋转头像",
    # 26. 红绿灯树 (测试语义冲突：树 vs 交通灯)
    # "Lanzhou Yellow River Mother Sculpture 兰州黄河母亲雕塑",
    # 27. 金属花 (测试开合瓣膜与金属质感)
    # "Floralis Genérica Buenos Aires 布宜诺斯艾利斯金属花",
    # 28. 气球狗 (测试“充气感”与光滑曲面)
    # "Giant Thermometer Chimney Shanghai 上海大烟囱温度计",
    # 29. 镂空地球仪 (测试经纬线结构与空心感)
    #  "Unisphere Queens New York 纽约法拉盛地球仪",
    # 30. 身体缺失者 (补位：极难，测试模型能否保留身体中空的特征而不自动补全)
    #  "Les Voyageurs sculpture Marseille 马赛旅行者雕塑"
    #"Songyue Pagoda 登封市嵩山嵩岳寺塔"
    #"Dole Plantation Maze 都乐种植园迷宫"
    #"青岛市五四广场五月的风雕塑"
    #"商丘市商字雕塑"
    #"Lanzhou Yellow River Iron Bridge 兰州黄河铁桥"
    #"长沙橘子洲青年毛泽东雕塑"
    "Tower Bridge of London 伦敦塔桥"
]


def run_baseline_experiment():
    experiment_log = []
    print(f"🚀 [Baseline] Starting Comparison Experiment...")

    for landmark_name in tqdm(LANDMARKS, desc="Baseline Processing"):
        try:
            print(f"\n🌍 Baseline 处理: {landmark_name} ...")

            # --- 修改开始：跳过 Grounder，单纯生成 ---
            # 1. 暂时不需要真值
            # spec = ground_entity_to_spec(landmark_name)
            # ref_url = ...

            # 2. 直接生成
            image_save_path = os.path.join(OUTPUT_BASE_DIR, f"{landmark_name}_baseline.png")
            prompt = f"An icon of {landmark_name}"

            success = generate_image_baseline(prompt, image_save_path)

            if success:
                print(f"   ✅ 图片生成完毕: {image_save_path}")

            # 3. 跳过评分
            # review_result = ...
            # --- 修改结束 ---

            time.sleep(1)

        except Exception as e:
            print(f"❌ Error: {e}")
            continue

    # 保存对比数据
    if experiment_log:
        df = pd.DataFrame(experiment_log)
        df.to_csv(CSV_SAVE_PATH, index=False, encoding='utf-8-sig')
        print(f"\n✅ Baseline 实验完成。数据保存至: {CSV_SAVE_PATH}")


if __name__ == "__main__":
    run_baseline_experiment()