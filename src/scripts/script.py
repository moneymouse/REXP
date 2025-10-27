import wandb
import argparse


parser = argparse.ArgumentParser(description='Delete wandb runs matching a keyword.')
parser.add_argument('-key', '--keyword', type=str, help='Keyword to match in run names') # 从命令行参数获取关键字
parser.add_argument('--exact', action='store_true', required=False, default=False, help='Use exact(`==`) matching instead of substring(`in`) matching') # 添加匹配类型参数
args = parser.parse_args()

# ==== 配置区域 ====
ENTITY = "zhihao_lin_SEU"    # 你的用户名或团队名
PROJECT = "alpha_02"  # 项目名
KEYWORD = args.keyword
# ==================

wandb.login()

api = wandb.Api()
runs = api.runs(f"{ENTITY}/{PROJECT}")

def main():
    # 找出所有匹配的 run
    if args.exact:
        # 精确匹配 - 完全匹配名称
        target_runs = [run for run in runs if KEYWORD.lower() == run.name.lower()]
    else:
        # 模糊匹配 - 关键字是名称的子字符串
        target_runs = [run for run in runs if KEYWORD.lower() in run.name.lower()]

    if not target_runs:
        print(f"没有找到包含关键字 '{KEYWORD}' 的 run。")
    else:
        print(f"找到 {len(target_runs)} 个 run 将被删除：")
        for run in target_runs:
            print(f"- {run.name} ({run.id})")   

        confirm = input("\n确认删除这些 run？输入 'y' 确认（默认'y'）：").strip().lower() or "y"
        if confirm != "y":
            print("❌ 已取消删除。")
        else:
            for run in target_runs:
                # 删除 run
                print(f"Deleting: {run.name} ({run.id})")
                run.delete()
            print("✅ 删除完成。") 

    

if __name__ == "__main__":
    main()