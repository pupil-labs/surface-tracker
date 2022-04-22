"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs
Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See LICENSE for license details.
---------------------------------------------------------------------------~(*)
"""


def test_package_metadata() -> None:
    import pupil_labs.surface_tracker as this_project

    assert hasattr(this_project, "__version__")
