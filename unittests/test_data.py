import pytest
import sys

sys.path.insert(1, "C:\\Users\\akash\\Documents\\MLProjects\\MLproject1\\model")

from data import Data


@pytest.mark.parametrize("batch_size, expected", [(100, 100), (100, 100)])
def test_data(batch_size, expected):
    d = Data(batch_size=batch_size)

    train, val = d.make_data()

    assert train.batch_size == expected
    assert len(train.dataset.data) == 50000
