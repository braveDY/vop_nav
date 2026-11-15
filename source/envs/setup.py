from setuptools import setup, find_packages
setup(name="envs", version="0.2.0", packages=["envs"] + ["envs." + p for p in find_packages(".")], package_dir={"envs": "."}, python_requires=">=3.10")
