from setuptools import setup, find_packages

setup(name="stage2", version="0.1.0",
      packages=["stage2"] + ["stage2." + p for p in find_packages(".")],
      package_dir={"stage2": "."}, python_requires=">=3.10",
      extras_require={"train": ["numpy", "torch"], "tensorboard": ["tensorboard"]})
