from dataclasses import dataclass

from typing import Any, Optional
import pandas as pd


@dataclass(frozen=True)
class FileInfo:
    name: str
    is_labeled: bool
    dimension: int
    file_name: str
    # delimiter: Optional[str] = None
    delimiter: Optional[str] = r"\s+"
    decimal: str = "."
    preprocess_func: Any = None

    def read_data(self):
        df = pd.read_csv(
            self.file_name, header=None, delimiter=self.delimiter, decimal=self.decimal
        )

        # print(str(self.delimiter))
        # print(self.name, df.info())

        if self.preprocess_func is not None:
            df = self.preprocess_func(df)

        labels = None
        if self.is_labeled:
            labels = df.pop(df.columns[self.dimension])  # pyright: ignore

        return df, labels


rp_features_count = 9


def preprocess_rp(df_original: pd.DataFrame):
    df = df_original.copy()
    df.iloc[:, :(rp_features_count)] -= 1
    df.iloc[:, rp_features_count] = df.iloc[:, rp_features_count].replace({2: 0, 4: 1})
    return df


files_info_list = []
files_info_list.extend(
    [
        FileInfo(
            "rp",
            True,
            rp_features_count,
            "../dane/rp.data",
            delimiter=r"\s+",
            decimal=",",
            preprocess_func=preprocess_rp,
        ),
        FileInfo("18Dunlabeled", False, 18, "../dane/data_18D.txt"),
    ]
)

two_dim_data = [
    FileInfo(f"2D_{i}", True, 2, f"../dane/dane_2D_{i}.txt") for i in range(1, 9)
]

# files_info_list.extend(two_dim_data)

files_info = {finf.name: finf for finf in files_info_list}
