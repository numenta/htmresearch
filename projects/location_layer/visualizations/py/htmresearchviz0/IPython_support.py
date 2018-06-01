import json
import numbers
import os
import uuid

from pkg_resources import resource_string

from IPython.display import HTML, display


def get_htmresearchviz0_js():
    path = os.path.join('package_data', 'htmresearchviz0-bundle.js')
    htmresearchviz0_js = resource_string('htmresearchviz0', path).decode('utf-8')
    return htmresearchviz0_js


def init_notebook_mode():
    style_inject = """
    <style>

    div.htmresearchviz-output {
      -webkit-touch-callout: none;
      -webkit-user-select: none;
      -moz-user-select: none;
      -ms-user-select: none;
      user-select: none;

      padding-bottom: 2px;
    }

    div.htmresearchviz-output svg {
      max-width: initial;
    }

    </style>
    """

    script_inject = u"""
    <script type='text/javascript'>
      if(!window.htmresearchviz0) {{
        define('htmresearchviz0', function(require, exports, module) {{
          {script}
        }});
        require(['htmresearchviz0'], function(htmresearchviz0) {{
          window.htmresearchviz0 = htmresearchviz0;
        }});
      }}
    </script>
    """.format(script=get_htmresearchviz0_js())

    display(HTML(style_inject + script_inject))


def printSingleLayer2DExperiment(csvText):
    elementId = str(uuid.uuid1())
    addChart = """
    <div class="htmresearchviz-output" id="%s"></div>
    <script>
    require(['htmresearchviz0'], function(htmresearchviz0) {
      htmresearchviz0.printRecording(document.getElementById('%s'), '%s');
    });
    </script>
    """ % (elementId, elementId,
           csvText.replace("\r", "\\r").replace("\n", "\\n"))

    display(HTML(addChart))


def printLocationModuleInference(logText):
    elementId = str(uuid.uuid1())
    addChart = """
    <div class="htmresearchviz-output" id="%s"></div>
    <script>
    require(['htmresearchviz0'], function(htmresearchviz0) {
      htmresearchviz0.locationModuleInference.printRecording(document.getElementById('%s'), '%s');
    });
    </script>
    """ % (elementId, elementId,
           logText.replace("\r", "\\r").replace("\n", "\\n"))

    display(HTML(addChart))


def printLocationModulesRecording(logText):
    elementId = str(uuid.uuid1())
    addChart = """
    <div class="htmresearchviz-output" id="%s"></div>
    <script>
    require(['htmresearchviz0'], function(htmresearchviz0) {
      htmresearchviz0.locationModules.printRecording(document.getElementById('%s'), '%s');
    });
    </script>
    """ % (elementId, elementId,
           logText.replace("\r", "\\r").replace("\n", "\\n"))

    display(HTML(addChart))


def printMultiColumnInferenceRecording(logText):
    elementId = str(uuid.uuid1())
    addChart = """
    <div class="htmresearchviz-output" id="%s"></div>
    <script>
    require(['htmresearchviz0'], function(htmresearchviz0) {
      htmresearchviz0.multiColumnInference.printRecording(document.getElementById('%s'), '%s');
    });
    </script>
    """ % (elementId, elementId,
           logText.replace("\r", "\\r").replace("\n", "\\n"))

    display(HTML(addChart))


def printPathIntegrationUnionNarrowingRecording(logText):
    elementId = str(uuid.uuid1())
    addChart = """
    <div class="htmresearchviz-output" id="%s"></div>
    <script>
    require(['htmresearchviz0'], function(htmresearchviz0) {
      htmresearchviz0.pathIntegrationUnionNarrowing.printRecording(document.getElementById('%s'), '%s');
    });
    </script>
    """ % (elementId, elementId,
           logText.replace("\r", "\\r").replace("\n", "\\n"))

    display(HTML(addChart))


def printSpikeRatesSnapshot(jsonText):
    elementId = str(uuid.uuid1())
    addChart = """
    <div class="htmresearchviz-output" id="%s"></div>
    <script>
    require(['htmresearchviz0'], function(htmresearchviz0) {
      htmresearchviz0.insertSpikeRatesSnapshot(document.getElementById('%s'), '%s');
    });
    </script>
    """ % (elementId, elementId,
           jsonText.replace("\r", "\\r").replace("\n", "\\n"))

    display(HTML(addChart))


def printSpikeRatesTimeline(jsonText):
    elementId = str(uuid.uuid1())
    addChart = """
    <div class="htmresearchviz-output" id="%s"></div>
    <script>
    require(['htmresearchviz0'], function(htmresearchviz0) {
      htmresearchviz0.insertSpikeRatesTimeline(document.getElementById('%s'), '%s');
    });
    </script>
    """ % (elementId, elementId,
           jsonText.replace("\r", "\\r").replace("\n", "\\n"))

    display(HTML(addChart))


def printInputWeights(jsonText):
    elementId = str(uuid.uuid1())
    addChart = """
    <div class="htmresearchviz-output" id="%s"></div>
    <script>
    require(['htmresearchviz0'], function(htmresearchviz0) {
      htmresearchviz0.insertInputWeights(document.getElementById('%s'), '%s');
    });
    </script>
    """ % (elementId, elementId,
           jsonText.replace("\r", "\\r").replace("\n", "\\n"))

    display(HTML(addChart))


def printOutputWeights(jsonText):
    elementId = str(uuid.uuid1())
    addChart = """
    <div class="htmresearchviz-output" id="%s"></div>
    <script>
    require(['htmresearchviz0'], function(htmresearchviz0) {
      htmresearchviz0.insertOutputWeights(document.getElementById('%s'), '%s');
    });
    </script>
    """ % (elementId, elementId,
           jsonText.replace("\r", "\\r").replace("\n", "\\n"))

    display(HTML(addChart))
