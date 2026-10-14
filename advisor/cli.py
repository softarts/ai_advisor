import argparse
import json
import logging
import os
import uuid
from pathlib import Path

from pa_analytics import Store
from dashboard import build_payload, HTML
from .agent import Advisor
from .tools import create_tools, portfolio_detail_tool, registry_tools


def _load_env():
    path = Path(".env")
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line=line.strip()
            if line and not line.startswith("#") and "=" in line:
                k,v=line.split("=",1); os.environ.setdefault(k.strip(),v.strip().strip('"\''))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--db",default="data/pa.db")
    parser.add_argument("--raw-dir",default="data/raw")
    parser.add_argument("--report",default="data/report_since_inception.plain")
    parser.add_argument("--user-id",default="default")
    args=parser.parse_args()
    _load_env()
    logdir=Path("logs"); logdir.mkdir(exist_ok=True)
    log_path=logdir/f"advisor_{uuid.uuid4().hex}.jsonl"
    logger=logging.getLogger("advisor"); logger.setLevel(logging.INFO)
    handler=logging.FileHandler(log_path,encoding="utf-8"); handler.setFormatter(logging.Formatter("%(message)s")); logger.addHandler(handler)
    store=Store(args.db,args.raw_dir)
    if all(os.environ.get(k) for k in ("LLM_API_KEY","LLM_BASE_URL","LLM_MODEL")):
        tools=create_tools(store)
        advisor=Advisor(tools,logger,args.user_id,
                        timezone_name=os.environ.get("APP_TIMEZONE", "Asia/Singapore"),
                        data_as_of=lambda: store.current().period_end)
    else: advisor=None
    thread_id=uuid.uuid4().hex
    print(f"AI Advisor · 日志 {log_path} · 输入 /help 查看命令")
    while True:
        try: line=input("你> ").strip()
        except (EOFError,KeyboardInterrupt): print(); break
        if not line: continue
        if line in ("/exit","/quit"): break
        if line=="/help":
            print("/import-report [文件路径] 导入报告；/history 本轮记忆；/clear 清空本轮；/remember 键=值 保存长期记忆；/recall 查看长期记忆；/exit 退出")
            continue
        if line.startswith("/import-report"):
            path=line[len("/import-report"):].strip() or args.report
            result=store.ingest(path)
            if result["action"] in ("added", "duplicate"):
                report=store.current()
                Path("dashboard.html").write_text(HTML.replace("__DATA__",json.dumps(build_payload(store,report),ensure_ascii=False)),encoding="utf-8")
                result["dashboard"]="dashboard.html"
            print(json.dumps(result,ensure_ascii=False,indent=2)); continue
        if line=="/clear":
            thread_id=uuid.uuid4().hex; print("已清空本轮对话记忆"); continue
        if line=="/history":
            state=advisor.graph.get_state({"configurable":{"thread_id":thread_id}}) if advisor else None
            print("暂无对话" if not state or not state.values else "\n".join(f"{m.type}: {m.content}" for m in state.values.get("messages",[])))
            continue
        if line.startswith("/remember "):
            key,sep,value=line[len("/remember "):].partition("=")
            if sep and key.strip():
                advisor.remember(key.strip(),value.strip())
                logger.info(json.dumps({"event":"memory_operation","operation":"put","user_id":args.user_id,"key":key.strip()},ensure_ascii=False))
                print("已保存到本次运行的长期记忆")
            else: print("格式：/remember 键=值")
            continue
        if line=="/recall":
            print(json.dumps(advisor.recall(),ensure_ascii=False,indent=2))
            logger.info(json.dumps({"event":"memory_operation","operation":"search","user_id":args.user_id},ensure_ascii=False))
            continue
        if not advisor:
            print("请在环境变量或 .env 中设置 LLM_API_KEY、LLM_BASE_URL、LLM_MODEL")
            continue
        trace=uuid.uuid4().hex
        logger.info(json.dumps({"event":"user_input","trace_id":trace,"input":line},ensure_ascii=False))
        try:
            answer=advisor.ask(line,thread_id,trace)
            logger.info(json.dumps({"event":"final_response","trace_id":trace,"response":answer},ensure_ascii=False))
            print(f"AI> {answer}")
        except Exception as e:
            logger.info(json.dumps({"event":"error","trace_id":trace,"error":str(e)},ensure_ascii=False))
            print(f"调用失败：{e}")


if __name__ == "__main__": main()
