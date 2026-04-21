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

# 导入模块
try:
    from Agent.agents.grounder_agent import ground_entity_to_spec
    from Agent.agents.designer_agent import run_designer, refine_designer
    from Agent.agents.reviewer_agent import run_reviewer
    from Agent.config import MODELS, OPENAI_API_KEY
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    sys.exit(1)

# ==========================================
# 📂 路径配置
# ==========================================
OUTPUT_BASE_DIR = r"Z:\python_projects\map_entropy\SymbolGeneration\Agent\outputs\images\Geo"
CSV_SAVE_PATH = r"Z:\python_projects\map_entropy\SymbolGeneration\Agent\outputs\images\Geo\1.29experiment_convergence_zh.csv"
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)


# ==========================================


def generate_image(prompt, save_path):
    """
    [黑白线稿 + 透明背景版] 强制生成背景透明的黑白符号
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    model_name = "gpt-image-1-mini"

    # --- 关键修改：风格控制 Prompt ---
    # 1. 移除了 "white background"
    # 2. 增加了 "transparent background", "PNG format", "isolated on transparent"

    style_suffix = (
        '''
        ", black and white line art icon, thick black outlines, transparent background, "
        "PNG format, isolated on transparent, flat 2D vector, no colors, no shading."
    '''
        ", flat vector illustration, app icon style, minimal vector graphics, "
        "white background, no shadows, no gradients, isolated on white."
    )


    # 组合 Prompt
    safe_prompt = prompt + style_suffix

    print(f"   🎨 Drawing with [{model_name}]: {safe_prompt[:40]}...")

    try:
        response = client.images.generate(
            model=model_name,
            prompt=safe_prompt,
            size="1024x1024",
            n=1
        )

        # 优先读取 b64_json
        image_data = None
        if hasattr(response.data[0], 'b64_json') and response.data[0].b64_json:
            image_data = base64.b64decode(response.data[0].b64_json)
            print(f"   ✅ Image received (Base64 default)")
        elif hasattr(response.data[0], 'url') and response.data[0].url:
            image_data = requests.get(response.data[0].url).content
            print(f"   ✅ Image received (URL)")

        if image_data:
            # 保存为 PNG 格式，以支持透明通道
            # 如果 save_path 结尾不是 .png，最好强制改一下，虽然不改也能存
            if not save_path.lower().endswith('.png'):
                save_path = os.path.splitext(save_path)[0] + ".png"

            with open(save_path, 'wb') as f:
                f.write(image_data)
            return True
        else:
            print("   ❌ Error: API response contained no image data.")
            return False

    except Exception as e:
        # 简单的安全重试机制
        if "safety" in str(e).lower() or "moderation" in str(e).lower():
            print("   ⚠️ 触发审核，尝试使用抽象几何描述重试...")
            # 重试 Prompt 也要记得改成透明背景
            retry_prompt = "Abstract geometric shape composed of black lines, transparent background, PNG format, simple symbol."
            try:
                resp = client.images.generate(model=model_name, prompt=retry_prompt, size="1024x1024", n=1)
                if hasattr(resp.data[0], 'b64_json') and resp.data[0].b64_json:
                    # 确保重试保存的文件名也是 png
                    if not save_path.lower().endswith('.png'):
                        save_path = os.path.splitext(save_path)[0] + ".png"
                    with open(save_path, 'wb') as f:
                        f.write(base64.b64decode(resp.data[0].b64_json))
                    return True
            except:
                pass

        print(f"   ❌ Generation Error: {e}")
        return False


# --- 精选 9 个代表性地标 (Pilot Run) ---
LANDMARKS = [

   # "Juscelino Kubitschek Bridge 儒塞利诺库比契克大桥",  # 英文定名，中文定域
 #   "Laguna Garzón Bridge 拉古纳加尔松桥",  # 强调“圆环”特征
 #   "Dragon Bridge Da Nang 越南岘港龙桥",  # 必须带“龙桥”二字，防止百度只搜到岘港的佛手桥

    # === 建筑类 ===
#    "Hallgrímskirkja Church 冰岛哈尔格林姆教堂",  # 加上 Church 增加语义权重
 #   "The Interlace Singapore 新加坡翠城新景",  # 加上 Singapore 确保地域准确
 #   "Heydar Aliyev Center 阿塞拜疆阿利耶夫中心",  # 双语全称
#
    # === 雕塑类 (最容易出错，需强约束) ===
   # "The Kelpies sculptures 苏格兰凯尔派马头",  # 加上“马头”二字，引导百度往马的方向搜
  #  "Sibelius Monument 芬兰西贝柳斯纪念碑",  # 标准双语
   # "Flamingo sculpture Chicago 芝加哥弗拉明戈雕塑" # 加上 City 和 Type

    # 1. [基准] 非对称拱桥 (Original)
    # "Gateshead Millennium Bridge 盖茨黑德千禧桥",
    # 2. DNA双螺旋 (极难，测试密集镂空)
    #"The Helix Bridge Singapore 新加坡双螺旋桥",
    # 3. 莫比乌斯环 (测试多层垂直交织)
    #"Lucky Knot Bridge Changsha 长沙幸运结桥",
    # 4. 蛇形流线 (测试有机形态)
    #"Python Bridge Amsterdam 阿姆斯特丹巨蟒桥",
    # 5. 波浪形拱 (测试扎哈风格的非对称流动感)
    #"Sheikh Zayed Bridge Abu Dhabi 谢赫扎耶德大桥",
    # 6. 木质波浪纹理 (测试材质与几何的结合)
    # "Henderson Waves Bridge Singapore 亨德森波浪桥",
    # 7. 垂直起伏如意形 (测试垂直方向的波浪，区别于平面圆环)
    # "Ruyi Bridge Shenxianju 神仙居如意桥",

    # ========================================================
    # Type II: Regular Architecture (规整建筑)
    # 测试点：透视畸变、几何组合、重复纹理、反直觉姿态
    # ========================================================
    # 1. [基准] 尖顶教堂 (Original - 强垂直线条)
    #"Wujin Lotus Conference Center 武进市民广场莲花馆"
    # 15. 蜂窝楼梯 (测试高频重复纹理与透视)
    ########################"Wuxi Taihu Show Theatre 无锡太湖秀剧场",

    #"Guangzhou Circle Mansion 广州圆大厦"

    # "Shenyang Coin Building 沈阳方圆大厦",
    # 3. 飞碟形状 (测试单点支撑平衡感)
    #"Niterói Contemporary Art Museum 尼泰罗伊当代艺术博物馆",
    # "Dole Plantation Maze 都乐种植园迷宫"
    #"Songyue Pagoda 登封市嵩山嵩岳寺塔"
    # 17. 球体+管道 (测试几何体连接关系)
    # "Atomium Brussels 布鲁塞尔原子塔",
    # 18. 网格+球体 (测试方正与圆形的强行嵌入)
    #"CCTV building 中央电视台大楼",
    # 19. 倒立建筑 (测试反直觉的姿态控制)
    # "WonderWorks Orlando 奥兰多颠倒屋",
    # 20. 倒金字塔/树形 (补位：测试头重脚轻的悬臂结构，区别于翠城)
    # "Geisel Library UCSD 盖泽尔图书馆",

    # ========================================================
    # Type III: Free-form Sculptures (自由形态雕塑)
    # 测试点：语义纠错、生物形态、材质抽象、局部与整体
    # ========================================================
    # 21. [基准] 局部马头 (Original - 测试生物特征放大)
    # "Maman sculpture Louise Bourgeois 毕尔巴鄂大蜘蛛雕塑",
    # 25. 切片人头 (测试旋转切片的对齐与错位)

    # 26. 红绿灯树 (测试语义冲突：树 vs 交通灯)
    #"Lanzhou Yellow River Mother Sculpture 兰州黄河母亲雕塑",
    # 27. 金属花 (测试开合瓣膜与金属质感)
    #"Floralis Genérica Buenos Aires 布宜诺斯艾利斯金属花",
    # 28. 气球狗 (测试“充气感”与光滑曲面)
    #"Giant Thermometer Chimney Shanghai 上海世博大烟囱温度计",
    # 29. 镂空地球仪 (测试经纬线结构与空心感)
    #"Unisphere Queens New York 纽约法拉盛地球仪",
    # 30. 身体缺失者 (补位：极难，测试模型能否保留身体中空的特征而不自动补全)
    # "Les Voyageurs sculpture Marseille 马赛旅行者雕塑身体缺失者"
    # "青岛市五四广场五月的风雕塑"
    #"商丘市商字雕塑"
    #"Aldar Headquarters Abu Dhabi 阿尔达总部大楼"
#"WonderWorks Orlando 奥兰多颠倒屋"
    #"Geisel Library UCSD 盖泽尔图书馆"
    #"Traffic Light Tree London 伦敦红绿灯树雕塑"
    #"Matrimandir Auroville India 黎明之城大金球"
   #"Prague Kafka bust Statue 布拉格卡夫卡头像雕像"
    #"Ali & Nino Statue in Georgia 格鲁吉亚的阿里和尼诺的雕像"
    #"Lanzhou Yellow River Iron Bridge 兰州黄河铁桥"
    #"广州五羊石像雕塑"
    "兰州白塔山"
]


def run_experiment_loop(max_rounds=3):
    experiment_log = []

    print(f"📂 Results will be saved to: {OUTPUT_BASE_DIR}")
    print(f"🚀 [Baidu Mode] Starting Convergence Experiment on {len(LANDMARKS)} landmarks...")

    for landmark_name in tqdm(LANDMARKS, desc="Processing"):
        try:
            print(f"\n🌍 正在处理: {landmark_name} ...")

            # --- Phase 1: Grounding ---
            # 增加重试机制，防止网络波动导致 VLM 失败
            spec = None
            for _ in range(3):
                try:
                    spec = ground_entity_to_spec(landmark_name)
                    if spec: break
                except Exception as e:
                    print(f"   ⚠️ Grounding retrying... ({e})")
                    time.sleep(2)

            if not spec:
                print("   ❌ Grounding failed after retries. Skipping.")
                continue

            ref_images = spec.get("reference_images", {})
            ref_url = ref_images.get("isometric") or ref_images.get("front") or ref_images.get("side")

            if not ref_url:
                print(f"⚠️ 警告: Grounder 未找到参考图，跳过...")
                continue

            structure_constraint = spec.get("structure", {})
            current_style_json = ""
            history_critiques = []

            # --- Phase 2: Iterative Loop ---
            for round_id in range(1, max_rounds + 1):
                print(f"  🔄 Round {round_id}...")
                image_save_path = os.path.join(OUTPUT_BASE_DIR, f"{landmark_name}_round_{round_id}.png")

                # 1. Designer
                if round_id == 1:
                    landmark_info = json.dumps({"name": landmark_name, "type": spec.get("entity_type", "landmark")},
                                               ensure_ascii=False)
                    current_style_json = run_designer(landmark_json=landmark_info, schema="{}", structure_spec=spec)
                else:
                    last_critique = history_critiques[-1] if history_critiques else {}
                    current_style_json = refine_designer(prev_style_json=current_style_json, review_data=last_critique,
                                                         structure_spec=spec)

                # 2. Generator
                try:
                    style_data = json.loads(current_style_json)
                    visual_prompt = style_data.get("visual_prompt_instruction", f"An icon of {landmark_name}")
                except:
                    visual_prompt = f"An icon of {landmark_name}"

                success = generate_image(visual_prompt, image_save_path)

                # 即使失败也记录数据，防止CSV为空
                if not success:
                    print("⚠️ 绘图失败，记录0分跳过...")
                    experiment_log.append({
                        "landmark": landmark_name,
                        "round": round_id,
                        "Structural Accuracy": 0,
                        "Style Consistency": 0,
                        "critique": "Generation Failed",
                        "image_path": "FAILED",
                        "ref_url": ref_url
                    })
                    continue

                # 3. Reviewer
                review_result = run_reviewer(
                    candidate_path=image_save_path,
                    reference_url=ref_url,
                    entity_name=landmark_name,
                    visual_instruction=json.dumps(structure_constraint, ensure_ascii=False)
                )

                history_critiques.append(review_result)

                # 4. Logging
                scores = review_result.get("scores", {})
                struct_score = scores.get("semantic_accuracy", 0)
                context = scores.get("contextual_consistency", 0)
                percept = scores.get("perceptual_clarity", 0)
                style_score = (context + percept) / 2.0

                experiment_log.append({
                    "landmark": landmark_name,
                    "round": round_id,
                    "Structural Accuracy": struct_score,
                    "Style Consistency": style_score,
                    "critique": review_result.get("critique", ""),
                    "image_path": image_save_path,
                    "ref_url": ref_url
                })

                print(f"    📊 得分: 结构={struct_score}, 风格={style_score}")
                time.sleep(1)

        except Exception as e:
            print(f"❌ 严重错误 {landmark_name}: {e}")
            continue

    if experiment_log:
        try:
            df = pd.DataFrame(experiment_log)
            df.to_csv(CSV_SAVE_PATH, index=False, encoding='utf-8-sig')
            print(f"\n✅ 数据已成功保存至: {CSV_SAVE_PATH}")
        except Exception as e:
            print(f"❌ 保存 CSV 失败: {e}")
    else:
        print("\n❌ 警告: 实验日志为空。")


if __name__ == "__main__":
    run_experiment_loop()