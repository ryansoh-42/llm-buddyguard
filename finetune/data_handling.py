from pathlib import Path

import datasets
import pandas as pd

from datasets import load_dataset


BASE_DATA_FILES: tuple[str] = (
    "data/all_bio_additional_filter.csv",
    "data/all_chem_additional_filter.csv",
    "data/all_physics_additional_filter.csv"
)

OUT_DATA_FILES: dict[str, str] = {
    "data/all_bio_additional_filter.csv": "data/bio.csv",
    "data/all_chem_additional_filter.csv": "data/chem.csv",
    "data/all_physics_additional_filter.csv": "data/physics.csv"
}


def convert_to_question_answer_format(filepath: str) -> None:
    assert filepath in OUT_DATA_FILES.keys(), f"Unexpected file: {filepath}"
    out_csv_path = OUT_DATA_FILES[filepath]

    if Path(out_csv_path).is_file():
        print("File already exists, exiting")
        return

    # Combine context_text and question_text into user message
    # Convert answer_text into final message
    df = pd.read_csv(filepath)

    df["user"] = df["context_text"] + df["question_text"]
    df["final"] = df["answer_text"]
    
    df = df.drop([
        "question_text", "context_text", "answer_text"
    ], axis=1)

    print(f"Writing out CSV to {out_csv_path}")
    df.to_csv(out_csv_path, index=False)


def convert_csv_to_dataset(filepath: str) -> datasets.Dataset:
    '''
    context_text, question_text, answer_text
    '''
    filepath_obj: Path = Path(filepath)

    assert filepath_obj.is_file(), f"{filepath} does not exist!"

    dataset = load_dataset("csv", data_files=filepath, split="train")

    return dataset


def main() -> None:
    # bio_dataset = convert_csv_to_dataset(BASE_DATA_FILES[0])

    # print(bio_dataset[0])
    for base_csv in BASE_DATA_FILES:
        convert_to_question_answer_format(base_csv)

    bio_dataset = convert_csv_to_dataset("data/bio.csv")
    print(bio_dataset[0])


if __name__ == '__main__':
    main()