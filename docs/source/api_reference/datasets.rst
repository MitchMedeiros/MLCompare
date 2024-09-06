Datasets
==============================

.. currentmodule:: mlcompare.data

The various Dataset classes primarily serve to turn a source of data into a Pandas DataFrame. Additionally, they 
pass any user provided parameters to the DatasetProcessor.

Classes
-------

.. autosummary::
   :toctree: generated/
   :template: custom_class.rst

   BaseDataset
   LocalDataset
   HuggingFaceDataset
   KaggleDataset
   OpenMLDataset
   DatasetFactory

BaseDataset
============

.. autoclass:: mlcompare.data.BaseDataset
   :members:
   :undoc-members:
   :show-inheritance:

LocalDataset
=============

.. autoclass:: mlcompare.data.LocalDataset
   :members:
   :undoc-members:
   :show-inheritance:

KaggleDataset
=============

.. autoclass:: mlcompare.data.KaggleDataset
   :members:
   :undoc-members:
   :show-inheritance:

HuggingFaceDataset
===================

.. autoclass:: mlcompare.data.HuggingFaceDataset
   :members:
   :undoc-members:
   :show-inheritance:

OpenMLDataset
=============

.. autoclass:: mlcompare.data.OpenMLDataset
   :members:
   :undoc-members:
   :show-inheritance:

DatasetFactory
===============
.. autoclass:: mlcompare.DatasetFactory
   :members:
   :undoc-members:
   :show-inheritance:
