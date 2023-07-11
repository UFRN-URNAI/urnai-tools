from setuptools import find_packages, setup


with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='urnai',
    packages=find_packages(),
    install_requires=[
    ],
    entry_points={
        'console_scripts': ['urnai=urnai.urnai_cmd:main'],
    },
    version='2.0-prev',
    description='A modular Deep Reinforcement Learning toolkit that supports multiple environments,'
                'such as PySC2, OpenAI Gym, ViZDoom and DeepRTS.',
    long_description=long_description,
    author='UFRN-IMD-URNAITeam',
    author_email='urnaiteam@gmail.com',
    url='https://github.com/UFRN-URNAI/urnai-tools',
)
