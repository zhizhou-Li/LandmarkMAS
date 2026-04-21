# -*- coding: utf-8 -*-
# 文件：Agent/agents/vectorizer_agent.py
import json
import os
from pathlib import Path
from utils import log, extract_json
from config import MODELS, OPENAI_API_KEY
from openai import OpenAI

# === 【修复1】修正导入路径和函数名 ===
from tools.run_color_vectorizer import process_clean_vectorization
from .semantic_vectorizer import semantic_vectorization_pipeline
from tools.check_topology import check_svg_topology

client = OpenAI(api_key=OPENAI_API_KEY)


class VectorizerAgent:
    def __init__(self, max_retries=3):
        self.max_retries = max_retries

    def run(self, image_path: str, final_spec: dict) -> str:
        log("VectorizerAgent", f"🚀 启动智能矢量化决策引擎...")

        # --- [第一阶段：LLM 自主决策提取路径] ---
        decision = self._decide_routing(image_path, final_spec)
        path_type = decision.get("path", "structure")
        # 补充默认的 min_area
        params = decision.get("initial_params", {"epsilon": 1.0, "n_clusters": 4, "min_area": 50})

        last_svg = None

        # --- [第二阶段：反思与动态调优闭环] ---
        for attempt in range(self.max_retries):
            log("VectorizerAgent", f"🌀 轮次 {attempt + 1}: 执行 {path_type} 路径, 参数: {params}")
            output_svg_path = image_path.replace(".png", f"_vec_v{attempt + 1}.svg")

            try:
                if path_type == "structure":
                    last_svg = semantic_vectorization_pipeline(
                        image_path=image_path,
                        output_svg_path=output_svg_path,
                        simplify_factor=params.get("epsilon", 1.0)
                    )
                else:
                    # 🧨 核心修复：彩色路径接收并使用大模型推演出的 min_area 参数
                    process_clean_vectorization(
                        input_path=image_path,
                        output_path=output_svg_path,
                        k=params.get("n_clusters", 4),
                        min_area=params.get("min_area", 50)
                    )
                    last_svg = output_svg_path
            except Exception as e:
                log("VectorizerAgent", f"⚠️ 矢量化执行出错: {e}")
                break

            if not last_svg or not os.path.exists(last_svg):
                log("VectorizerAgent", "⚠️ SVG 未成功生成。")
                break

            total_polys, error_count, has_error = check_svg_topology(last_svg)

            if not has_error:
                log("VectorizerAgent", "✅ 拓扑检查 100% 通过，数据已达到 Analysis-Ready 标准。")
                break

            log("VectorizerAgent", f"⚠️ 检测到 {error_count} 处拓扑异常，呼叫大模型重新推演参数...")
            if attempt < self.max_retries - 1:
                # 🧨 核心修复：传入 path_type，让 LLM 知道它在调什么类型的图
                params = self._reflect_and_adjust(params, error_count, path_type)

        return last_svg

    def _decide_routing(self, image_path: str, spec: dict) -> dict:
        """【真·智能体】调用 LLM 依据语义意图决策提取路径和初始参数"""
        system_prompt = """
        You are an expert GIS Algorithm Engineer. Your task is to select the optimal vectorization pipeline for a map symbol based on its design specification.

        Available pipelines:
        1. "structure": Best for minimalist, black-and-white, highly geometric, or simple line-art icons (e.g., flat bridge silhouettes).
        2. "color": Best for complex, multi-colored, illustrative, or styled symbols.

        Rules:
        - Analyze the user's specification (entity_type, constraints, palette).
        - Output strictly in JSON format.
        - Default epsilon (Douglas-Peucker simplification tolerance) is usually 1.0 to 2.0.
        - If 'color' is chosen, infer a reasonable 'n_clusters' (usually 3 to 6) based on the design's complexity.

        JSON Schema:
        {
            "path": "structure" | "color",
            "initial_params": {
                "epsilon": float,
                "n_clusters": int (only needed if path is color)
                "min_area": int
            },
            "reason": "brief explanation"
        }
        """

        try:
            resp = client.chat.completions.create(
                model=MODELS["LLM_MODEL"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Design Specification:\n{json.dumps(spec, ensure_ascii=False)}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.1  # 保持较低温度以保证稳定性
            )
            content = resp.choices[0].message.content
            decision = extract_json(content)
            log("VectorizerAgent_Routing", decision)

            if not decision or "path" not in decision:
                return {"path": "color", "initial_params": {"n_clusters": 4, "epsilon": 1.0}}
            return decision

        except Exception as e:
            log("VectorizerAgent_Routing_Error", str(e))
            return {"path": "color", "initial_params": {"n_clusters": 4, "epsilon": 1.0}}

    def _reflect_and_adjust(self, prev_params: dict, error_count: int, path_type: str) -> dict:
        """【真·智能体】调用 LLM 依据拓扑错误报告，像人类一样推演并调整参数"""
        system_prompt = """
        You are a Topology Optimization Agent. The previous vectorization attempt yielded topological errors (e.g., self-intersecting polygons).
        Your goal is to adjust the algorithm hyperparameters to eliminate these errors.

        Rules:
        - The primary parameter to fix self-intersections is "epsilon" (Douglas-Peucker tolerance). 
        - Increasing "epsilon" simplifies the geometry, reducing vertices and eliminating complex self-intersections.
        - If the error count is high (>10), make a bolder adjustment (e.g., +1.0 or +1.5).
        - If the error count is low (<3), make a minor tweak (e.g., +0.5).
        - Output strictly in JSON format.

        JSON Schema:
        {
            "epsilon": float (the new updated value),
            "n_clusters": int (keep the previous value if it exists),
            "reason": "Explain why you adjusted the epsilon by this specific amount."
        }
        """

        user_content = f"""
                Previous Parameters: {json.dumps(prev_params)}
                Topology Report: Found {error_count} self-intersecting polygons.
                Please provide the new parameters.
                """

        try:
            resp = client.chat.completions.create(
                model=MODELS["LLM_MODEL"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            new_params_full = extract_json(resp.choices[0].message.content)
            log("VectorizerAgent_Reflect", new_params_full)

            # 更新参数
            new_params = prev_params.copy()
            if "epsilon" in new_params_full: new_params["epsilon"] = new_params_full["epsilon"]
            if "n_clusters" in new_params_full: new_params["n_clusters"] = new_params_full["n_clusters"]
            if "min_area" in new_params_full: new_params["min_area"] = new_params_full["min_area"]

            return new_params

        except Exception as e:
            log("VectorizerAgent_Reflect_Error", str(e))
            new_params = prev_params.copy()
            if path_type == "structure":
                new_params["epsilon"] = new_params.get("epsilon", 1.0) + 0.5
            else:
                new_params["min_area"] = new_params.get("min_area", 50) + 30
            return new_params

        except Exception as e:
            log("VectorizerAgent_Reflect_Error", str(e))
            new_params = prev_params.copy()
            new_params["epsilon"] = new_params.get("epsilon", 1.0) + 0.5
            return new_params


def run_vectorizer_agent(image_path: str, final_spec: dict):
    agent = VectorizerAgent()
    return agent.run(image_path, final_spec)