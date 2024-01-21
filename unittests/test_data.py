import pytest
import os
import sys

PATH = os.getcwd() + '/model'

print(PATH)

sys.path.append(PATH)

from data import Data

@pytest.mark.parametrize("batch_size, expected", [(100, 100), (100, 100)])
def test_data(batch_size, expected):
    d = Data(batch_size=batch_size)

    train, val = d.make_data()

    assert train.batch_size == expected
    assert len(train.dataset.data) == 50000