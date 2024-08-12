from setuptools import setup, find_packages

setup(
    name="wikisearch",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        # Add dependencies here
    ],
    entry_points={
        "console_scripts": [
            "wikisearch=wikisearch.cli:main",
        ],
    },
)
