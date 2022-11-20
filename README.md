# QN-S3VM BFGS optimizer for semi-supervised support vector machines.

This implementation provides both a L-BFGS optimization scheme for semi-supvised support vector machines. Details can be found in:

```{.bib}
@InProceeding{Gieseke:2012
    authors = {F. Gieseke and A. Airola and T. Pahikkala and O. Kramer},
    title = {Sparse quasi-Newton optimization for semi-supervised support vector machines},
    booktitle = {the 1st Int. Conf. on Pattern Recognition Applications and Methods},
    year = 2012,
    pages = {45--54}
}
```

## modification from [NekoYIQ/QNS3VM](https://www.github.com/NekoYIQ/QNS3VM)


# RUNNING THE EXAMPLES

For a description of the data sets, see the paper mentioned above and the references therein. Running the command "python qns3vm.py" should yield an output similar to:

| data set instance | # of labeled patterns | # of unlabeled patterns | # of test patterns | Time needed to compute the model in sec. | Classification error of QN-S3VM |
|---|---|---|---|---|---|
|Sparse text|48|924|974|0.775886058807|0.0667351129363|
|Dense gaussian|25|225|250|0.464584112167|0.012|
|Dense moons|5|495|500|0.69714307785|0.0|
