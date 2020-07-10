from setuptools import setup, find_packages
import sys

if not (sys.version.startswith('3.5') or sys.version.startswith('3.6')):
    raise Exception('Only Python 3.5 and 3.6 are supported')

setup(name='rlcodebase',
      packages=[package for package in find_packages()
                if package.startswith('rlcodebase')],
      install_requires=[
        torch==1.4.0
        gym>=0.10.8
        torchvision==0.5.0
        opencv-python
        pandas
        pathlib
        numpy],
      dependency_links=['git+git://github.com/openai/baselines.git@8e56dd#egg=baselines']
      description="Codebase of Reinforcement Learning Algorithms",
      author="Jinwei Xing",
      url='https://github.com/KarlXing/RLCodebaseL',
      author_email="jinweixing1006@gmail.com",
      version="0.1")