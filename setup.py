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
    'version': '0.1.26',
    'description': '',
    'long_description': '# QN-S3VM BFGS optimizer for semi-supervised support vector machines.\n\nThis implementation provides both a L-BFGS optimization scheme for semi-supvised support vector machines. Details can be found in:\n\n```{.bib}\n@InProceeding{Gieseke:2012\n    authors = {F. Gieseke and A. Airola and T. Pahikkala and O. Kramer},\n    title = {Sparse quasi-Newton optimization for semi-supervised support vector machines},\n    booktitle = {the 1st Int. Conf. on Pattern Recognition Applications and Methods},\n    year = 2012,\n    pages = {45--54}\n}\n```\n\n## modification from [NekoYIQ/QNS3VM](https://github.com/NekoYIQI/QNS3VM) or [tmadl/semisup-learn](https://github.com/tmadl/semisup-learn)\n\n### `class qns3vm.QN_S3VM(X_l, L_l, X_u, lam, lamU, sigma, kernel_type, estimate_r)`\n\n`L_l` must labeled 2-classes\n\n### `class qns3vm.QN_S3VM_OVR(X_l, L_l, X_u, lam, lamU, sigma, kernel_type, estimate_r)`\n\n### `qns3vm.datasets`\n\ndata acquiring functions. these are from original example\'s main code.\n\n- `X_train_l, L_train_l, X_train_u, X_test, L_test = get_moons_data()`\n- `X_train_l, L_train_l, X_train_u, X_test, L_test = get_text_data()`\n- `X_train_l, L_train_l, X_train_u, X_test, L_test = get_gaussian_data()`\n\n### `qns3vm.tools`\n\n- `plot_distribution(ax,clf,X_train_l, L_train_l, X_test, L_test)->None`\n- `classification_error(preds, L_test)->float`\n\n# RUNNING THE EXAMPLES\n\nFor a description of the data sets, see the paper mentioned above and the references therein.\n\n| data set instance | # of labeled patterns | # of unlabeled patterns | # of test patterns | Time needed to compute the model in sec. | Classification error of QN-S3VM |\n|---|---|---|---|---|---|\n|Sparse text|48|924|974|0.775886058807|0.0667351129363|\n|Dense gaussian|25|225|250|0.464584112167|0.012|\n|Dense moons|5|495|500|0.69714307785|0.0|\n\n```{python}\nfrom qns3vm import QN_S3vm\nfrom qns3vm.datasets import get_moons_data\nfrom matplotlib import pyplot as plt\n\nX_train_l, L_train_l, X_train_u, X_test, L_test = get_moons_data()\nclf = QN_S3VM(X_train_l, L_train_l, X_train_u, lam=, lamU=, sigma=, kernel_type="RBF", estimate_r=0)\nclf.train()\npreds = clf.predict(X_test)\nerror = classification_error(preds,L_test)\nprint(f"classification error of QN-S3VM: {error}")\n\nfigure = plt.figure()\nax = figure.add_subplot(1,1,1)\nplot_distribution(ax, clf, X_train_l, L_train_l, X_test, L_test)\n```\n',
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
