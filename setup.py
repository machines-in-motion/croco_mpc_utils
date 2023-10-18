from setuptools import setup
from os import path

package_name = 'croco_mpc_utils'

        
setup(
    name=package_name,
    version="1.0.0",
    package_dir={package_name: path.join('.')},
    packages=[package_name],
    install_requires=["setuptools", 
                      "importlib_resources"],
    zip_safe=True,
    maintainer="skleff",
    maintainer_email="sk8001@nyu.edu",
    long_description_content_type="text/markdown",
    url="https://github.com/machines-in-motion/croco_mpc_utils",
    description="Utilities for easy & modular MPC prototyping using Crocoddyl.",
    license="BSD-3-clause",
    entry_points={
        "console_scripts": [],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD-3-clause",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)