from __future__ import annotations

import sarFbr


def testVersionIsSet():
    assert isinstance(sarFbr.__version__, str)
    assert sarFbr.__version__
