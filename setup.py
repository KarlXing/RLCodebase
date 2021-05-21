from setuptools import setup, find_packages
import sys

if not (sys.version.startswith('3')):
    raise Exception('RLCodebase supports Python3 only.')

setup(name='rlcodebase',
      packages=[package for package in find_packages()
                if package.startswith('rlcodebase')],
      install_requires=[
        'torch==1.4.0',
        'tensorflow==2.5.0',
        'gym>=0.10.8',
        'torchvision==0.5.0',
        'opencv-python',
        'pandas',
        'pathlib',
        'numpy'],
      description="Codebase of Reinforcement Learning Algorithms",
      author="Jinwei Xing",
      url='https://github.com/KarlXing/RLCodebase',
      author_email="jinweixing1006@gmail.com",
      version="0.1")