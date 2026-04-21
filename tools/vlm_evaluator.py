# -*- coding: utf-8 -*-
import os
import base64
import json
import pandas as pd
from openai import OpenAI

# ================= 配置区域 =================
# 您的 OpenAI API Key
api_key = "sk-your-openai-api-key"
client = OpenAI(api_key=api_key)

# 假设您的文件夹结构如下：
# dataset/
# ├── GT/ (存放真实地标照片，如 kelpies_gt.jpg)
# ├── BaselineA/ (存放 A 生成的图片，如 kelpies.png)
# ├── BaselineB/ (存放 B 生成的图片)
# └── Ours/ (存放 Ours 生成的图片)
BASE_DIR = r"Z:\python_projects\map_entropy\SymbolGeneration\evaluation"
METHODS = ["BaselineA", "BaselineB", "Ours"]


# ============================================

def encode_image(image_path):
    """将本地图片转换为 Base64 编码格式供 API 读取"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def evaluate_symbol_with_vlm(gt_path, generated_path, landmark_name):
    """调用 GPT-4o 作为专家审稿人进行打分"""
    gt_base64 = encode_image(gt_path)
    gen_base64 = encode_image(generated_path)

    system_prompt = """
    You are an expert Cartographer and a rigorous Reviewer for the 'Cartography and Geographic Information Science' journal.
    Your task is to evaluate a generated map symbol against a real-world Ground Truth (GT) photograph of a landmark.

    You must evaluate TWO metrics:
    1. 'semantic_pass' (0 or 1): Does the generated symbol correctly identify and represent the core macro-structure/posture of the GT landmark? 
       - 1 (Pass): Correct identity and basic structure.
       - 0 (Fail): Severe hallucination (e.g., wrong posture, missing iconic components like towers/arches).
    2. 'structural_score' (1-10): How topologically accurate and geometrically refined is the symbol?
       - 1-3: Broken, excessive noise, unusable.
       - 4-6: Basic shape, but topologically flawed (e.g., fused parts, missing minor features).
       - 7-8: Good structure, but slight stylistic artifacts.
       - 9-10: Perfect geometric abstraction, highly accurate topology, ready for GIS use.

    Output strictly in JSON format: {"semantic_pass": int, "structural_score": int, "reason": "brief explanation"}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            temperature=0.0,  # 必须设为0，保证学术评估的确定性与可复现性
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": f"Evaluate this generated map symbol for the landmark: {landmark_name}."},
                    {"type": "text", "text": "Image 1: Ground Truth Photograph"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{gt_base64}"}},
                    {"type": "text", "text": "Image 2: Generated Map Symbol"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{gen_base64}"}}
                ]}
            ],
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        print(f"Evaluation error for {landmark_name}: {e}")
        return {"semantic_pass": 0, "structural_score": 0, "reason": "API Error"}


def main():
    print("🚀 启动 VLM 自动化制图学评估协议...")
    results = []

    # 获取 GT 文件夹下的所有测试地标名称
    gt_dir = os.path.join(BASE_DIR, "GT")
    if not os.path.exists(gt_dir):
        print("❌ 找不到 GT 文件夹，请配置正确的路径。")
        return

    landmarks = [f for f in os.listdir(gt_dir) if f.endswith(('.png', '.jpg'))]

    for filename in landmarks:
        landmark_name = os.path.splitext(filename)[0]
        gt_path = os.path.join(gt_dir, filename)

        for method in METHODS:
            gen_path = os.path.join(BASE_DIR, method, filename)

            if not os.path.exists(gen_path):
                continue  # 如果某方法没有生成这张图则跳过

            print(f"🧐 正在评估: {landmark_name} | 方法: {method}")
            eval_data = evaluate_symbol_with_vlm(gt_path, gen_path, landmark_name)

            results.append({
                "Landmark": landmark_name,
                "Method": method,
                "Semantic_Pass": eval_data.get("semantic_pass", 0),
                "Structural_Score": eval_data.get("structural_score", 0),
                "Reason": eval_data.get("reason", "")
            })

    # 将结果汇总为 DataFrame
    df = pd.DataFrame(results)

    # 打印消融实验核心指标总结 (Table 5-2 的数据来源)
    print("\n" + "=" * 50)
    print("📊 消融实验定量评估结果 (Ablation Study Metrics)")
    print("=" * 50)
    summary = df.groupby("Method").agg(
        Semantic_Success_Rate=('Semantic_Pass', lambda x: f"{(x.mean() * 100):.1f}%"),
        Avg_Structural_Score=('Structural_Score', 'mean')
    ).reset_index()
    print(summary.to_string(index=False))

    # 保存详细报告
    output_csv = os.path.join(BASE_DIR, "ablation_evaluation_results.csv")
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n✅ 详细评分报告已保存至: {output_csv}")


if __name__ == "__main__":
    main()