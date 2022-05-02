import os
from setuptools import find_packages, setup

setup(
    name='urnai',
    packages=find_packages(),
    install_requires=[
        'absl-py',
        'gym',
        'tensorflow',
        'numpy',
        'matplotlib',
        'pysc2',
        'pandas',
        'psutil',
        'GPUtil',
    ],
    dependency_links=dep_links,
    entry_points={
        'console_scripts': ['urnai=urnai.urnai_cmd:main'],
    },
    version='1.0',
    description='A modular Deep Reinforcement Learning toolkit that supports multiple environments,'
                'such as PySC2, OpenAI Gym, ViZDoom and DeepRTS.',
    long_description='URNAI Tools is a modular Deep Reinforcement Learning (DRL) toolkit'
                     'that supports multiple environments, such as PySC2,'
                     'OpenAI Gym, ViZDoom and DeepRTS. The main goal of URNAI'
                     'Tools is to provide an easy-to-use modular platform'
                     'for the development of DRL agents. Each part of a'
                     'typical Reinforcement Learning scenario, such as'
                     'the environment, the learning algorithm,'
                     'the action space and so on, is considered'
                     'a module in URNAI and can simply be swaped.'
                     'Beyond that, it supplies a series of out-of-the-box DRL algorithms,'
                     'environment wrappers, action wrappers, reward'
                     'functions and state representations,'
                     'allowing developers to easily assemble different learning'
                     'configurations and quickly iterate through them.',
    author='UFRN-URNAI',
    author_email='urnaiteam@gmail.com',
    url='https://github.com/UFRN-URNAI/URNAI-Tools',
)
