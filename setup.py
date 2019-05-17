from setuptools import setup, find_packages

VERSION = 0.1

setup(name='deep_reinforcement_learning_algorithms_with_pytorch',
      version=VERSION,
      description='',
      url='https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch.git',
      author='Petros Christodoulou',
      author_email='p.christodoulou2@gmail.com',
      license='',
      packages=[package for package in find_packages()],
      zip_safe=False,
      # install_requires=[ # Still deciding what extra libraries are needed. Up to Petros.
      #     'pybullet>=1.7.8', 'gym', 'numpy', 'torch', 'tensorboardX', 'namedlist'
      # ],
)