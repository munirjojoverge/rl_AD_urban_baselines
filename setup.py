from setuptools import setup, find_packages

setup(
    name='urban_AD-env',
    version='1.0.dev0',
    description='An environment for simulated urban_AD driving tasks',
    url='https://github.com/eleurent/urban_AD-env',
    author='Edouard Leurent',
    author_email='eleurent@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Researchers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.7',
    ],

    keywords='autonomous urban_AD driving simulation environment reinforcement learning',
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

