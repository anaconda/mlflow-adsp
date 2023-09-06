import setuptools
import versioneer
from setuptools import setup

setup(
    name="mlflow-adsp",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="MLFlow Plugin For The Anaconda Data Science Platform",
    packages=setuptools.find_packages(),
    author="Joshua C. Burt",
    python_requires=">=3.8",
    install_requires=["mlflow>=2.3.0", "ae5-tools>=0.6.1", "psutil", "pydantic<2.0", "tqdm", "click", "requests"],
    entry_points={
        # Define a MLFlow Project Backend plugin called 'adsp'
        "mlflow.project_backend": "adsp=mlflow_adsp:adsp_backend_builder"
    },
    scripts=["mlflow_adsp/bin/mlflow-adsp"],
)
