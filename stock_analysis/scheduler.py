"""
台股每日篩選排程器

使用方式：
    python -m stock_analysis.scheduler --mode apscheduler   # APScheduler 常駐
    python -m stock_analysis.scheduler --mode cron          # 列印 crontab 指令
    python -m stock_analysis.scheduler --run-now            # 立即執行一次
    python -m stock_analysis.scheduler --run-now --output-dir ./my_output
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


def _run_now(output_dir: str, fallback_file: str) -> None:
    from .screener import run_screener_job
    run_screener_job(output_dir=output_dir, fallback_file=fallback_file)


def _start_apscheduler(output_dir: str, fallback_file: str, hour: int = 18, minute: int = 0) -> None:
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
    except ImportError:
        logger.error("請先安裝 apscheduler：pip install apscheduler")
        sys.exit(1)

    from .screener import run_screener_job

    scheduler = BlockingScheduler(timezone="Asia/Taipei")
    scheduler.add_job(
        run_screener_job,
        trigger="cron",
        day_of_week="mon-fri",
        hour=hour,
        minute=minute,
        kwargs={"output_dir": output_dir, "fallback_file": fallback_file},
    )

    print(f"APScheduler 排程已啟動")
    print(f"  執行時間：週一至週五 {hour:02d}:{minute:02d} 台北時間")
    print(f"  輸出目錄：{os.path.abspath(output_dir)}")
    print("  按 Ctrl+C 停止\n")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\n排程已停止")


def _print_crontab(output_dir: str, fallback_file: str) -> None:
    script_dir = os.path.abspath(os.path.dirname(__file__) + "/..")
    cron_line = (
        f"0 18 * * 1-5  cd {script_dir} && "
        f"python -m stock_analysis.scheduler --run-now "
        f"--output-dir {output_dir} --fallback-file {fallback_file}"
    )
    print("\n請將以下行加入 crontab（執行 crontab -e）：\n")
    print(cron_line)
    print("\n說明：每週一至週五 18:00 自動執行（台股 13:30 收盤後）\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="台股每日篩選排程器")
    parser.add_argument("--mode", choices=["apscheduler", "cron"],
                        help="排程模式：apscheduler 常駐 / cron 列印指令")
    parser.add_argument("--run-now", action="store_true", help="立即執行一次篩選")
    parser.add_argument("--output-dir", default="./screener_output", help="CSV 輸出目錄")
    parser.add_argument("--fallback-file", default="tickers.txt",
                        help="TWSE 抓取失敗時的備用股票清單檔案")
    parser.add_argument("--hour", type=int, default=18, help="APScheduler 排程小時（預設 18）")
    parser.add_argument("--minute", type=int, default=0, help="APScheduler 排程分鐘（預設 0）")
    args = parser.parse_args()

    if args.run_now:
        _run_now(args.output_dir, args.fallback_file)
    elif args.mode == "apscheduler":
        _start_apscheduler(args.output_dir, args.fallback_file, args.hour, args.minute)
    elif args.mode == "cron":
        _print_crontab(args.output_dir, args.fallback_file)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
