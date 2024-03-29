import os

from setuptools import find_packages, setup


def is_optional_enabled(optional):
    return os.environ.get(optional, None) is not None


# Dependencies versions
VERSION_ABSL = '>=1.2.0'
VERSION_GYM = '>=0.26.1'
VERSION_TF = '>=2.6.0'
VERSION_NUMPY = '>=1.23.0'
VERSION_MATPLOTLIB = '>=3.6.0'
VERSION_KERAS = '>=2.6.0'
VERSION_PYSC2 = '>=3.0.0'
VERSION_PANDAS = '>=1.5.0'
VERSION_PSUTIL = '>=5.9.2'

VERSION_VIZDOOM = ''
VERSION_DRTS = 'stable'

VIZDOOM = 'URNAI_VIZDOOM'
TF_CPU = 'URNAI_TF_CPU'
DEEPRTS = 'URNAI_DEEPRTS'
INSTALL_KERAS_FROM_REPO = 'URNAI_INSTALL_KERAS_FROM_REPO'
LATEST_DEPS = 'URNAI_LATEST_DEPS'

git_url = '{package} @ git+https://github.com/{user}/{repo}@{branch}#egg={package}'
dep_links = []
dep_list = []
tf = 'tensorflow-gpu' + VERSION_TF

if is_optional_enabled(LATEST_DEPS):
    print('Dependencies will be installed in theirs latest versions.')
    VERSION_ABSL = ''
    VERSION_GYM = ''
    VERSION_TF = ''
    VERSION_NUMPY = ''
    VERSION_MATPLOTLIB = ''
    VERSION_KERAS = ''
    VERSION_PYSC2 = ''
    VERSION_PANDAS = ''
    VERSION_PSUTIL = ''
    VERSION_VIZDOOM = ''


if is_optional_enabled(DEEPRTS):
    print('DeepRTS installation enabled.')
    dep_list.append(git_url.format(user='marcocspc',
                                   repo='deep-rts',
                                   branch=VERSION_DRTS,
                                   package='deeprts'))

if is_optional_enabled(VIZDOOM):
    print('VizDoom installation enabled.')
    dep_list.append('vizdoom' + VERSION_VIZDOOM)

if is_optional_enabled(TF_CPU):
    print('Tensorflow cpu will be installed instead of Tensorflow GPU.')
    tf = 'tensorflow' + VERSION_TF

if is_optional_enabled(INSTALL_KERAS_FROM_REPO):
    print('Keras from official repo will be installed instead of Tensorflow built-in.')
    dep_list.append('keras' + VERSION_KERAS)

setup(
    name='urnai',
    packages=find_packages(),
    install_requires=[
        'absl-py' + VERSION_ABSL,
        'gym' + VERSION_GYM,
        'protobuf<3.20,>=3.9.2',
        tf,
        'numpy' + VERSION_NUMPY,
        'matplotlib' + VERSION_MATPLOTLIB,
        'pysc2' + VERSION_PYSC2,
        'pandas' + VERSION_PANDAS,
        'psutil' + VERSION_PSUTIL,
        'GPUtil',
    ] + dep_list,
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
    author='UFRN-IMD-URNAITeam',
    author_email='urnaiteam@gmail.com',
    url='https://github.com/marcocspc/URNAI-Tools',
)
