from distutils.core import setup

setup(
    name="featuretweaking",
    version="0.1",
    packages=["featuretweaking"],
    install_requires=["numpy", "xgboost", "pandas", "matplotlib", "pytest", "scikit-learn"],
)
