import numpy as np

from firescape import mtbs


def test_dnbr6_to_barc4_semantics():
    dnbr6 = np.array([[0, 1, 2, 3], [4, 5, 6, 3]])
    barc = mtbs.dnbr6_to_barc4(dnbr6)
    # 1..4 pass through; background(0), greening(5), non-mapping(6) -> nodata 0
    assert barc.tolist() == [[0, 1, 2, 3], [4, 0, 0, 3]]
    assert barc.dtype == np.uint8


def test_greening_and_mask_never_burned():
    barc = mtbs.dnbr6_to_barc4(np.array([5, 6]))
    assert (barc == 0).all()
