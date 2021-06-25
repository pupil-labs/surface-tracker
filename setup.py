"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs
Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See LICENSE for license details.
---------------------------------------------------------------------------~(*)
"""
from setuptools import setup

setup(
    extras_require={
        "dev": ["pre-commit", "tox"],
        "deploy": ["build", "twine", "bump2version"],
        "example": [
            "opencv-python",
            "pupil-apriltags",
            "matplotlib",
            "Pillow",
            "msgpack",
        ],
    }
)
