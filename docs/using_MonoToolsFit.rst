Using MonoTools.fit
-------------

.. role:: python(code)
   :language: python

There are two ways of interacting with the :py:obj:`everest` catalog: via the command line and
through the Python interface. For quick visualization, check out the :doc:`everest <everest>` and
:doc:`estats <estats>` command line tools.
For customized access to de-trended light curves in Python, keep reading!

.. contents::
   :local:

A Simple Example
================

Once you've installed :py:obj:`everest`, you can easily import it in Python:

.. code-block :: python

   import everest

Say we're interested in **EPIC 201367065**, a :py:obj:`K2` transiting exoplanet host star.
Let's instantiate the :py:class:`Everest <everest.user.Everest>` class for this target:

.. code-block :: python

   star = everest.Everest(201367065)

You will see the following printed to the screen:

.. code-block :: bash