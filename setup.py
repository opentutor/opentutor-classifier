from os import path
from setuptools import setup, find_packages


this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


def _read_dependencies():
    requirements_file = "requirements.txt"
    with open(requirements_file) as fin:
        return [line.strip() for line in fin if line]


packages = find_packages()
requirements = _read_dependencies()


setup(
    name="opentutor_classifier",
    version="1.0.0",
    author_email="vinitbodhwani123@gmail.com",
    description="train and run inference for open tutor",
    packages=packages,
    package_dir={"opentutor_classifier": "opentutor_classifier"},
    package_data={"opentutor_classifier": ["py.typed"]},
    scripts=["bin/opentutor_classifier"],
    install_requires=requirements,
    long_description=long_description,
    long_description_content_type="text/markdown",
)
