# minimal_codex_test.py
# 目标：触发 Codex 插件激活并输出诊断信息

import os
import sys

def main():
    print("=== Codex Test Script ===")
    print("Python version:", sys.version)
    print("Current file:", __file__)
    print("Working directory:", os.getcwd())

    # 测试代码块
    # Codex 插件通常在函数内或注释触发
    # 在这里写一条注释触发 AI 生成
    # Generate docstring for this function
    def add(a, b):
        return a + b

    # 手动输出提示信息，用于确认插件是否可以访问
    print("Test: Function 'add' defined, Codex should see this if activated.")

if __name__ == "__main__":
    main()