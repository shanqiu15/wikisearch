import argparse
import os
import pprint
from .query_service import QueryService

qs = QueryService()


def main():
    parser = argparse.ArgumentParser(description="Wiki QA CLI Tool")
    parser.add_argument("query", type=str, help="Query to search on Wikipedia")

    args = parser.parse_args()

    if "OPENAI_API_KEY" in os.environ:
        print("Answer: ", qs.answer(args.query))
    else:
        pprint.pprint(qs.search(args.query), width=40, indent=4)


if __name__ == "__main__":
    main()
