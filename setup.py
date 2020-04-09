"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs
Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See LICENSE for license details.
---------------------------------------------------------------------------~(*)
"""
from setuptools import find_packages, setup


package_dir = "src"
package = "surface_tracker"


install_requires = ["numpy"]  # TODO: Add OpenCV to requirements


with open("README.md") as f:
    readme = f.read()


with open("CHANGELOG.md") as f:
    changelog = f.read()


long_description = f"{readme}\n\n{changelog}"


if __name__ == "__main__":
    setup(
        author="Pupil Labs GmbH",
        author_email="pypi@pupil-labs.com",
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Natural Language :: English",
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering",
        ],
        description="Surface tracker",
        extras_require={"dev": ["pytest", "tox"]},
        install_requires=install_requires,
        long_description=long_description,
        long_description_content_type="text/markdown",
        name="surface-tracker",
        packages=find_packages(package_dir),
        package_dir={"": package_dir},
        url="https://github.com/pupil-labs/surface-tracker",
        version="0.0.1",
        zip_save=False,
    )
