from setuptools import setup, find_packages
setup(name="assets", version="0.2.0", packages=["assets"] + ["assets." + p for p in find_packages(".")], package_dir={"assets": "."}, python_requires=">=3.10")
