# "Location Layer" Visualizations

This folder contains:

- JavaScript visualizations
- A `package.json` that creates a single js file which contains the visualization code and all dependencies
- A `htmresearchviz0` Python package that lets you print these visualizations in a Jupyter notebook.

## How to use

~~~
npm install
npm run build

cd py/
python setup.py develop --user

cd ..
npm run dev
~~~

This final command watches for changes to .js files, and it also launches a
webserver where you can use the various log viewers.

If you want to use Jupyter notebook, include:

~~~python
import htmresearchviz0.IPython_support as viz

# Include this once at the top of the notebook.
viz.init_notebook_mode()
~~~

Now you're free to call other functions in the `IPython_support` module.
