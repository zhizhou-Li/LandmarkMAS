# -*- coding: utf-8 -*-
# 文件：Agent/graph_orchestrator.py
from core.state import SymbolGenerationState
from agents.interpreter_agent import run_interpreter
from agents.grounder_agent import ground_entity_to_spec
from agents.spec_infer_agent import infer_spec
from agents.designer_agent import run_designer, refine_designer
from agents.generator_agent import run_generator
from agents.reviewer_agent import run_reviewer
from agents.vectorizer_agent import VectorizerAgent
import json


class LandmarkGraphNodes:
    """封装所有图节点"""

    @staticmethod
    def node_cognition(state: SymbolGenerationState) -> dict:
        print("\n--- [Node] 认知与视觉锚定 (Grounding) ---")
        user_input = state["user_input"]

        intent_schema = run_interpreter(user_input)
        try:
            intent_data = json.loads(intent_schema) if isinstance(intent_schema, str) else intent_schema
            entity_name = intent_data.get("entity", {}).get("name", user_input)
        except:
            entity_name = user_input

        # 调用改造后的 ReAct Grounder
        grounder_spec = ground_entity_to_spec(user_input, search_focus=entity_name)
        final_spec = infer_spec(user_input, grounder_spec)

        ref_images = grounder_spec.get("reference_images", {})
        ref_url = ref_images.get("isometric") or ref_images.get("front") or grounder_spec.get("reference_image_url")

        vlm_fact_str = ""
        if "vlm_analysis" in grounder_spec:
            v = grounder_spec["vlm_analysis"]
            vlm_fact_str = f"Posture: {v.get('posture')}, Shape: {v.get('shape_description')}"

        return {
            "intent_schema": intent_schema, "entity_name": entity_name,
            "grounder_spec": grounder_spec, "final_spec": final_spec,
            "ref_url": ref_url, "vlm_fact_str": vlm_fact_str
        }

    @staticmethod
    def node_design(state: SymbolGenerationState) -> dict:
        print(f"\n--- [Node] 样式设计 (Round {state['round_idx'] + 1}) ---")
        if state.get("critique"):
            print("📝 Designer 正在根据 Reviewer 的批评修改 JSON...")
            style_json = refine_designer(
                prev_style_json=state["current_style_json"],
                review_data={"critique": state["critique"]},
                structure_spec=state["final_spec"]
            )
        else:
            style_json = run_designer(state["entity_name"], state["intent_schema"], state["final_spec"])

        return {"current_style_json": style_json, "round_idx": state["round_idx"] + 1}

    @staticmethod
    def node_generate(state: SymbolGenerationState) -> dict:
        print("\n--- [Node] 候选图像生成 ---")
        paths = run_generator(
            outline_path=None, style_json=state["current_style_json"],
            user_text=state["entity_name"], structure_spec=state["final_spec"]
        )
        return {"candidate_paths": paths}

    @staticmethod
    def node_review(state: SymbolGenerationState) -> dict:
        print("\n--- [Node] VLM 四维评估 (含客观工具检测) ---")
        best_path = None
        best_acc = -1
        best_ctx = -1  # [新增] 记录上下文一致性最高分
        best_review = {}

        for path in state["candidate_paths"]:
            review = run_reviewer(
                candidate_path=path, reference_url=state["ref_url"],
                entity_name=state["entity_name"], visual_instruction=state["vlm_fact_str"]
            )

            # [修正] 同时获取两个维度的分数
            scores = review.get("scores", {})
            acc = scores.get("semantic_accuracy", 0)
            ctx = scores.get("contextual_consistency", 0)

            # [修正] 只有当总分更高时，才更新最佳候选（避免高准确度但完全无视颜色的结果成为最佳）
            if (acc + ctx) > (best_acc + best_ctx):
                best_acc = acc
                best_ctx = ctx
                best_path = path
                best_review = review

        return {
            "best_candidate_path": best_path,
            "acc_score": best_acc,
            "ctx_score": best_ctx,  # [新增] 将一致性分数加入状态流转
            "decision": best_review.get("decision", "FAIL"),
            "critique": best_review.get("critique", "Improve structure and colors.")
        }

    @staticmethod
    def node_vectorize(state: SymbolGenerationState) -> dict:
        print("\n--- [Node] 智能矢量化与拓扑重建 ---")
        agent = VectorizerAgent()
        svg_path = agent.run(state["best_candidate_path"], state["final_spec"])
        return {"final_svg_path": svg_path}


class LandmarkGraphWorkflow:
    def __init__(self, max_rounds=5, required_accuracy=8):
        self.max_rounds = max_rounds
        self.required_accuracy = required_accuracy
        self.nodes = LandmarkGraphNodes()

    def run(self, user_input: str) -> SymbolGenerationState:
        state: SymbolGenerationState = {
            "user_input": user_input, "intent_schema": None, "entity_name": "",
            "grounder_spec": None, "final_spec": None, "vlm_fact_str": "", "ref_url": "",
            "current_style_json": None, "round_idx": 0, "candidate_paths": [],
            "best_candidate_path": None, "critique": None, "decision": None, "acc_score": 0,
            "final_svg_path": None
        }

        # 1. 启动认知层
        state.update(self.nodes.node_cognition(state))

        # 2. Actor-Critic 设计与辩论内循环
        debate_rounds = 0
        while debate_rounds < self.max_rounds:
            # Designer 设计
            state.update(self.nodes.node_design(state))
            # Generator 绘图
            state.update(self.nodes.node_generate(state))
            # Reviewer 审查
            state.update(self.nodes.node_review(state))
            print(
                f"\n👨‍⚖️ [Debate Result] 决策: {state['decision']}, 准确度: {state['acc_score']}, 规则一致性: {state.get('ctx_score', 0)}")

            # [修正] 必须满足决策为PASS，且【准确度】和【规则一致性】双双达标，才能跳出循环
            if state["decision"] == "PASS" and state["acc_score"] >= self.required_accuracy and state.get("ctx_score",
                                                                                                          0) >= self.required_accuracy:
                print("🤝 辩论通过，Designer 与 Reviewer 达成完美的制图共识！")
                break
            else:
                print("🥊 辩论未通过（形似度或制图规则未达标）：Reviewer 驳回了设计。准备进入下一轮修改...")
                debate_rounds += 1

        if debate_rounds >= self.max_rounds:
            print("🛑 达到最大辩论轮数，强制采纳当前最佳结果。")

        # 3. 流转至矢量化拓扑重建
        state.update(self.nodes.node_vectorize(state))
        return state