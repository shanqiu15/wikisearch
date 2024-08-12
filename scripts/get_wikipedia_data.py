from datasets import load_dataset

SNAPSHOT_DATE = "20240720"


def get_wikipedia_data():
    ds = load_dataset("olm/wikipedia", language="en", date=SNAPSHOT_DATE)
    ds.save_to_disk("./wikipedia")


if __name__ == "__main__":
    get_wikipedia_data()
