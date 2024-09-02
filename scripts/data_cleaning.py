import pandas as pd
import os

cur_path = os.path.dirname(__file__)
new_path = os.path.relpath('../data/raw.csv', cur_path)


def generate_diff_columns(col_name: str, dataframe: pd.DataFrame) -> pd.DataFrame:
    diff_col: pd.DataFrame = (dataframe[f'TotalPost{col_name}']
                              - dataframe[f'TotalPre{col_name}'])
    return diff_col


def main() -> pd.DataFrame:
    raw_df: pd.DataFrame = pd.read_csv(new_path)

    for col_name in ("HADS", "CDS", "PSQI"):
        raw_df[f'{col_name}_diff'] = generate_diff_columns(
            col_name=col_name, dataframe=raw_df)

    return raw_df


def generate_data() -> pd.DataFrame:
    return main()


if __name__ == "__main__":
    main()
