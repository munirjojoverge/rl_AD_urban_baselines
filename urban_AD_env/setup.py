######################################################################
#          Deep Reinforcement Learning for Autonomous Driving
#                  Created/Modified on: January 10, 2019
#                      Author: Munir Jojo-Verge
#######################################################################

from setuptools import setup, find_packages

setup(
    name='urban_AD_env',
    version='1.0.dev0',
    description='An environment for simulated urban driving tasks',
    url='',
    author='Munir Jojo-Verge (Based on Edouard Leurent work: https://github.com/eleurent/highway-env)',
    author_email='munirjojoverge@yahoo.es',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Researchers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
    ],

    keywords='autonomous urban driving simulation environment reinforcement learning',
    packages=find_packages(exclude=['docs', 'scripts', 'tests']),
    install_requires=['gym', 'numpy', 'pygame', 'jupyter', 'matplotlib', 'pandas'],
    tests_require=['pytest'],
    extras_require={
        'dev': ['scipy'],
        'deploy': ['pytest-runner', 'sphinx<1.7.3', 'sphinx_rtd_theme']
    },
    entry_points={
        'console_scripts': [],
    },
)

