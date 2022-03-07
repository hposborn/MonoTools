# Installing MonoTools
*********

## Pip installation
`MonoTools` is installable via pip, so the following should work: `pip install monotools`

## Installing direct from GitHub
Alternatively, to run the most up-to-date development version, you can run `git clone http://github.com/hposborn/MonoTools`, cd into the `MonoTools` folder, then run `pip install .` (plus make sure the folder where MonoTools is installed is included in your \$PYTHONPATH, e.g. by adding `export PYTHONPATH=/path/to/dir:\$PYTHONPATH` to your .bashrc file).

## The \$MONOTOOLSDIR environment variable
The default location to store files is within the installed `MonoTools` package (i.e. MonoTools/MonoTools/data). However, this can be modified with the environment variable `$MONOTOOLSDIR` (e.g. by placing `export MONOTOOLSDIR="/path/to/new/folder/"` in your `.bashrc` file).

Mac OSX users may need to make sure PyMC3, exoplanet and theano are all properly installed, which require GCC (e.g. using brew install gcc) and C libraries are present.

Be aware that `monotools` only works with python 3 and will almost certainly break on windows.
