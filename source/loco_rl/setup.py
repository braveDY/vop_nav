from setuptools import setup, find_packages
setup(name="loco-rl", version="0.2.0", packages=["loco_rl"] + ["loco_rl." + p for p in find_packages(".")], package_dir={"loco_rl": "."}, python_requires=">=3.10")
