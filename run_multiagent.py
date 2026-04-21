# -*- coding: utf-8 -*-
# SymbolGeneration/Agent/run_multiagent.py

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
from graph_orchestrator import LandmarkGraphWorkflow
# --- 【关键修复】路径注入 ---
# 这段代码确保无论你在哪里运行脚本，Python 都能找到 'Agent' 包
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent  # SymbolGeneration 目录
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
# ---------------------------

import time
import json

# --- 环境变量配置 (保留你的代理设置) ---
os.environ["http_proxy"] = "http://127.0.0.1:7897"
os.environ["https_proxy"] = "http://127.0.0.1:7897"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# --- Agent 导入 ---
# 注意：这里假设你的 Agent 代码都已经按照之前的讨论修改完毕
from agents.interpreter_agent import run_interpreter
from agents.grounder_agent import ground_entity_to_spec
from agents.spec_infer_agent import infer_spec
from agents.designer_agent import run_designer, refine_designer
from agents.generator_agent import run_generator
from agents.reviewer_agent import run_reviewer
from agents.vectorizer_agent import run_vectorizer_agent

# --- 工具导入 ---
from config import CREATIVE_SAMPLES
from utils import save_json

# ------------------------------------------------------------------------------
# 闭环控制配置
# ------------------------------------------------------------------------------
MAX_ROUNDS = 3  # 最大迭代轮数
PASS_THRESHOLD = 7.0  # 平均分及格线 (0-10)
REQUIRED_ACCURACY = 8  # "准确表达" (Semantic Accuracy) 必须达到的分数




def main():
    user_query = "生成卡通风格的兰州老街的符号，要求结构特征可辨、色彩协调"

    # 实例化图架构工作流
    workflow = LandmarkGraphWorkflow(max_rounds=5, required_accuracy=8)

    # 启动流转
    final_state = workflow.run(user_query)
    print(f"🎉 任务圆满结束！最终矢量文件：{final_state.get('final_svg_path')}")


if __name__ == "__main__":
    main()