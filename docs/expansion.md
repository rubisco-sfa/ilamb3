# Expand ILAMB

This package has been written with the intention that the community can expand it in a number of ways:

1. **Add reference datasets:** If you have or know of a reference dataset that is directly comparable to model output, then expanding ILAMB can be as easy as encoding/ensuring the dataset follows the [CF-Conventions](https://cfconventions.org/). While we plan to have tutorials for how this can be done, for the moment we direct you to this [example](https://github.com/rubisco-sfa/ilamb3-data/blob/main/data/WECANN/convert.py) which automatically downloads the raw source data, changes how time is encoded, and raises the meta-data quality to meet standards.

2. **Expand the transform library:** The ILAMB analysis routines are written to compare two sources which are directly comparable (that is, the same variable). Yet sometimes, the reference dataset does not directly compare to anything that models output. For example, the [NSIDC](https://nsidc.org/data/GGD318/versions/2) dataset estimates the extent of permafrost but this is not currently modeled explicitly in global climate models. We implement a `permafrost` transform that takes the model's soil temperature data and estimates permafrost extent using published methods. It may be that the comparison you wish to make requires you to [implement an ILAMB transform](transform) function.

3. **Expand the analysis library:** ILAMB implements a set of general routines meant to compare and synthesis the performance of variables in a number of dimensions: bias, RMSE, annual cycle, spatial distribution, relationships and more. It may be that these techniques are not sufficient to measure the performance in the way you wish. In this case, you will need to [implement an ILAMB analysis](analysis) function.
