
Documentation for ilamb3
========================

A rewrite of ILAMB has been a long time in the works. The ecosystem of scientific python libraries has changed dramatically since we first wrote ILAMB. Much of the software we wrote to understand the `CF`_ conventions is now more completely and elegantly handled by `xarray`_ and related packages.

Originally we wrote ILAMB to function like a replacement to the diagnostic packages that modeling centers run--a holistic analysis over large amounts of model output. However, since then we have seen an increased demand from users to also run parts ILAMB analyses in their own scripts and notebooks. As this was not a use case for which we originally designed, it was quite difficult and we ended up writing a lot of custom code to meet users' needs.

We are building the new ILAMB from the bottom up, documenting and releasing as we go. This is in part because a full rewrite is a lot of work and this strategy allow users to work with what we have completed to this point. It also is a way for us to communicate with the community for feedback to help hone the package design. Eventually the goal is that this package will replace the current `ILAMB`_ package.

Design Principles
-----------------

As development continues, we will update this list of design principles which guide ilamb3 developments.

1. The ILAMB analysis methods should be more modular and operate on xarray datasets. Our original implementation made adding datasets easy, but the analysis itself was quite challenging to expand. It is our goal to make adding an analysis method more simple and our basic object be the xarray dataset which the user is more likely to understand.
2. The user should be able to import individual analysis functions and run locally in their scripts and notebooks.

Installation
------------

This package is being developed and *not* currently listed in PyPI or conda-forge. You may install it directly from the github repository::

  pip install git+https://github.com/rubisco-sfa/ilamb3

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: I want to...

   jupyter
   analysis

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Methods

   prelim
   bias
   relationship
   nbp

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Reference

   api

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Communicate

   GitHub repository <https://github.com/rubisco-sfa/ilamb3>
   Bugs or Suggestions <https://github.com/rubisco-sfa/ilamb3/issues>

.. _xarray: https://docs.xarray.dev/en/stable/
.. _CF: https://cfconventions.org/
.. _ILAMB: https://github.com/rubisco-sfa/ILAMB
