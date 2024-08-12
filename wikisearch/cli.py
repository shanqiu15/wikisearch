import argparse
import os
from .query_service import QueryService
from rich.console import Console
from rich.pretty import pprint

qs = QueryService()


def main():
    parser = argparse.ArgumentParser(description="Wiki QA CLI Tool")
    parser.add_argument("query", type=str, help="Query to search on Wikipedia")

    args = parser.parse_args()

    console = Console()

    if "OPENAI_API_KEY" in os.environ:
        pprint(qs.answer(args.query), console=console)
    else:
        pprint(qs.search(args.query), console=console)


if __name__ == "__main__":
    main()
