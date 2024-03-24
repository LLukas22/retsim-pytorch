from src.retsim_pytorch.preprocessing import binarize


def test_binarize():
    txts = ["hello", "world"]
    arr, num_chunks = binarize(txts)
    assert arr.shape == (2, 512, 24)
    assert len(num_chunks) == 2
