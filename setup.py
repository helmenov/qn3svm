# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['qns3vm', 'qns3vm.datasets']

package_data = \
{'': ['*']}

install_requires = \
['matplotlib>=3.6.1,<4.0.0',
 'numpy>=1.23.5,<2.0.0',
 'pandas>=1.5.1,<2.0.0',
 'scipy>=1.9.3,<2.0.0']

setup_kwargs = {
    'name': 'qns3vm',
    'version': '0.1.0',
    'description': '',
    'long_description': '# QN-S3VM BFGS optimizer for semi-supervised support vector machines.\n\nThis implementation provides both a L-BFGS optimization scheme for semi-supvised support vector machines. Details can be found in:\n\n```{.bib}\n@InProceeding{Gieseke:2012\n    authors = {F. Gieseke and A. Airola and T. Pahikkala and O. Kramer},\n    title = {Sparse quasi-Newton optimization for semi-supervised support vector machines},\n    booktitle = {the 1st Int. Conf. on Pattern Recognition Applications and Methods},\n    year = 2012,\n    pages = {45--54}\n}\n```\n\n## modification from [NekoYIQ/QNS3VM](https://www.github.com/NekoYIQ/QNS3VM)\n\n\n# RUNNING THE EXAMPLES\n\nFor a description of the data sets, see the paper mentioned above and the references therein. Running the command "python qns3vm.py" should yield an output similar to:\n\n| data set instance | # of labeled patterns | # of unlabeled patterns | # of test patterns | Time needed to compute the model in sec. | Classification error of QN-S3VM |\n|---|---|---|---|---|---|\n|Sparse text|48|924|974|0.775886058807|0.0667351129363|\n|Dense gaussian|25|225|250|0.464584112167|0.012|\n|Dense moons|5|495|500|0.69714307785|0.0|\n',
    'author': 'Kotaro SONODA',
    'author_email': 'kotaro1976@gmail.com',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.10,<3.11',
}


setup(**setup_kwargs)
