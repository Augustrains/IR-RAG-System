import os
import json
import random
from typing import List, Dict

random.seed(42)

# =========================
# 路径配置
# =========================

# 负样本文本：每行一个问题
NEGATIVE_TEXT_PATH = "/root/autodl-tmp/IR-RAG-System/data/ut/raw_general_chats.txt"

# 你当前已有的 Alpaca 训练/测试集
TRAIN_ALPACA_PATH = "/root/autodl-tmp/IR-RAG-System/data/qa_pairs/train_augmented_alpaca.jsonl"
TEST_ALPACA_PATH = "/root/autodl-tmp/IR-RAG-System/data/qa_pairs/test_augmented_alpaca.jsonl"

# 输出：拆分后的负样本
NEG_TRAIN_ALPACA_PATH = "/root/autodl-tmp/IR-RAG-System/data/qa_pairs/negative_train_alpaca.jsonl"
NEG_TEST_ALPACA_PATH = "/root/autodl-tmp/IR-RAG-System/data/qa_pairs/negative_test_alpaca.jsonl"

# 输出：最终合并后的训练/测试集
FINAL_TRAIN_ALPACA_PATH = "/root/autodl-tmp/IR-RAG-System/data/qa_pairs/train_augmented_with_neg_alpaca.jsonl"
FINAL_TEST_ALPACA_PATH = "/root/autodl-tmp/IR-RAG-System/data/qa_pairs/test_augmented_with_neg_alpaca.jsonl"

SYSTEM_PROMPT = "你是信息检索领域的专业助手。"
NEGATIVE_INSTRUCTION = "请判断用户问题是否属于当前知识范围。如果问题与当前知识范围无关，或无法根据当前知识回答，请输出“无答案”。"


# =========================
# 基础工具
# =========================

def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    if not os.path.exists(path):
        print(f"[warn] file not found: {path}")
        return rows

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"[load jsonl error] line={line_no}, err={e}")
    return rows


def save_jsonl(path: str, rows: List[Dict]):
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# =========================
# 读取负样本文本
# =========================

def load_negative_questions(path: str) -> List[str]:
    questions = []
    if not os.path.exists(path):
        print(f"[warn] negative file not found: {path}")
        return questions

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            q = normalize_text(line)
            if not q:
                continue
            questions.append(q)

    return questions


def dedup_questions(questions: List[str]) -> List[str]:
    seen = set()
    out = []
    for q in questions:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


# =========================
# 9:1 切分
# =========================

def split_train_test(questions: List[str], train_ratio: float = 0.9, seed: int = 42):
    items = questions[:]
    random.Random(seed).shuffle(items)

    train_size = int(len(items) * train_ratio)
    train_questions = items[:train_size]
    test_questions = items[train_size:]
    return train_questions, test_questions


# =========================
# 转 Alpaca
# =========================

def build_negative_alpaca_rows(questions: List[str]) -> List[Dict]:
    rows = []
    for q in questions:
        row = {
            "instruction": NEGATIVE_INSTRUCTION,
            "input": q,
            "output": "无答案",
            "system": SYSTEM_PROMPT
        }
        rows.append(row)
    return rows


# =========================
# 合并数据集
# =========================

def merge_datasets(base_rows: List[Dict], neg_rows: List[Dict], shuffle: bool = True, seed: int = 42) -> List[Dict]:
    merged = base_rows + neg_rows
    if shuffle:
        random.Random(seed).shuffle(merged)
    return merged


# =========================
# 主流程
# =========================

def main():
    # 1. 读取负样本
    neg_questions = load_negative_questions(NEGATIVE_TEXT_PATH)
    print("负样本原始条数：", len(neg_questions))

    # 2. 去重
    neg_questions = dedup_questions(neg_questions)
    print("负样本去重后条数：", len(neg_questions))

    # 3. 9:1 切分
    neg_train_questions, neg_test_questions = split_train_test(
        neg_questions,
        train_ratio=0.9,
        seed=42
    )
    print("负样本训练集条数：", len(neg_train_questions))
    print("负样本测试集条数：", len(neg_test_questions))

    # 4. 转 Alpaca
    neg_train_rows = build_negative_alpaca_rows(neg_train_questions)
    neg_test_rows = build_negative_alpaca_rows(neg_test_questions)

    # 5. 保存拆分后的负样本
    save_jsonl(NEG_TRAIN_ALPACA_PATH, neg_train_rows)
    save_jsonl(NEG_TEST_ALPACA_PATH, neg_test_rows)

    # 6. 读取原训练/测试集
    train_rows = load_jsonl(TRAIN_ALPACA_PATH)
    test_rows = load_jsonl(TEST_ALPACA_PATH)

    print("原训练集条数：", len(train_rows))
    print("原测试集条数：", len(test_rows))

    # 7. 并入对应训练/测试集
    final_train_rows = merge_datasets(train_rows, neg_train_rows, shuffle=True, seed=42)
    final_test_rows = merge_datasets(test_rows, neg_test_rows, shuffle=True, seed=42)

    print("加入负样本后的训练集条数：", len(final_train_rows))
    print("加入负样本后的测试集条数：", len(final_test_rows))

    # 8. 保存最终文件
    save_jsonl(FINAL_TRAIN_ALPACA_PATH, final_train_rows)
    save_jsonl(FINAL_TEST_ALPACA_PATH, final_test_rows)

    print("负样本训练集保存路径：", NEG_TRAIN_ALPACA_PATH)
    print("负样本测试集保存路径：", NEG_TEST_ALPACA_PATH)
    print("最终训练集保存路径：", FINAL_TRAIN_ALPACA_PATH)
    print("最终测试集保存路径：", FINAL_TEST_ALPACA_PATH)

    # 9. 预览
    if neg_train_rows:
        print("\n===== 负样本训练样例 =====")
        print(json.dumps(neg_train_rows[0], ensure_ascii=False, indent=2))

    if final_train_rows:
        print("\n===== 合并后训练集样例 =====")
        print(json.dumps(final_train_rows[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
