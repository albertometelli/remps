import sys

from setuptools import find_packages, setup

if sys.version_info.major != 3:
    print(
        "This Python is only compatible with Python 3, but you are running "
        "Python {}. The installation will likely fail.".format(sys.version_info.major)
    )


setup(
    name="remps",
    packages=[package for package in find_packages() if package.startswith("remps")],
    install_requires=[
        "gym[classic_control]",
        "scipy",
        "tqdm",
        "joblib",
        "zmq",
        "dill",
        "progressbar2",
        "mpi4py",
        "cloudpickle",
        "tensorflow>=1.4.0",
        "click",
        "opencv-python",
        "numpy",
        "tensorflow",
        "theano",
    ],
    description="Remps",
    author="EmanueleGhelfi, Alberto Maria Metelli, Marcello Restelli",
    author_email="albertomariameterlli@polimi.it, emanuele.ghelfi@mail.polimi.it",
    version="0.0.1",
)
