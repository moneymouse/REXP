import os
import requests
import datetime
import wandb
import argparse
from pathlib import Path

home = Path.home()

def save_notion_credentials():
    notion_token = input("Please input Notion Token:")
    database_id = input("Please input Database ID:")
    with open(home / ".rexp_secret", "a+") as f:
        f.write(f"NOTION_TOKEN={notion_token}\n")
        f.write(f"LOG_DATABASE_ID={database_id}\n")

if not Path(home / ".rexp_secret").exists():
    save_notion_credentials()

with open(home / ".rexp_secret", "r") as f:
    for line in f:
        key, value = line.strip().split("=")
        os.environ[key] = value

# 环境变量中存放 Notion 信息
NOTION_TOKEN = os.getenv("NOTION_TOKEN", "")
DATABASE_ID = os.getenv("LOG_DATABASE_ID", "")

if not NOTION_TOKEN or not DATABASE_ID:
    save_notion_credentials()

def sync_run_to_notion(runs):
    latest_run = runs  # 获取最近一次 run

    name = f"{latest_run.name}_{latest_run.id}"
    url = latest_run.url
    acc = latest_run.summary.get("acc", None)
    loss = latest_run.summary.get("loss", None)

    notion_headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28"
    }
    data = {
        "parent": {"database_id": DATABASE_ID},
        "properties": {
            "Experiment": {"title": [{"text": {"content": name}}]},
            "Wandb URL": {"url": url},
            "Accuracy": {"number": acc},
            "Loss": {"number": loss},
            "Timestamp": {"date": {"start": datetime.datetime.utcnow().isoformat()}}
        }
    }

    r = requests.post("https://api.notion.com/v1/pages", headers=notion_headers, json=data)
    r.raise_for_status()
    print(f"✅ Synced W&B run '{name}' to Notion!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp", "--experiment_id", help="The experiment id used to identify the experiment. \n " \
                    "Recommend to set EXP_ID environmental variable.", type=str)
    args = parser.parse_args()
    project = os.getenv("EXP_ID", args.experiment_id)

    if not project:
        print("❌ Please set the experiment id via -exp or EXP_ID environment variable.")
        return
    
    try:
        api = wandb.Api()
        last_run = api.runs(project)[0]
        sync_run_to_notion(last_run)
    except Exception as e:
        print(f"❌ Failed to sync W&B run to Notion: {e}")

if __name__ == "__main__":
    main()
