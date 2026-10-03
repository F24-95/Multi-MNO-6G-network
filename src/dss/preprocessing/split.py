import pandas as pd
from typing import Tuple

TRAIN_SPLIT_MAX = 2879
VAL_SPLIT_MAX = 3455
TEST_SPLIT_MAX = 4031

def tag_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tags each row with its chronological split:
    - train: steps 0 to 2879 (days 1-10)
    - val: steps 2880 to 3455 (days 11-12)
    - test: steps 3456 to 4031 (days 13-14)
    """
    tagged = df.copy()
    conditions = [
        tagged["step_idx"] <= TRAIN_SPLIT_MAX,
        (tagged["step_idx"] > TRAIN_SPLIT_MAX) & (tagged["step_idx"] <= VAL_SPLIT_MAX),
        tagged["step_idx"] > VAL_SPLIT_MAX,
    ]
    choices = ["train", "val", "test"]
    tagged["split"] = pd.Series(pd.Categorical.from_codes(
        pd.Series(0, index=tagged.index).where(conditions[0],
            pd.Series(1, index=tagged.index).where(conditions[1], 2)
        ),
        categories=choices
    ))
    return tagged

def get_split_subsets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits DataFrame into (train_df, val_df, test_df) based on step_idx."""
    train_df = df[df["step_idx"] <= TRAIN_SPLIT_MAX].copy()
    val_df = df[(df["step_idx"] > TRAIN_SPLIT_MAX) & (df["step_idx"] <= VAL_SPLIT_MAX)].copy()
    test_df = df[df["step_idx"] > VAL_SPLIT_MAX].copy()
    return train_df, val_df, test_df
