## HTM Research with Natural Language Processing (NLP)
=============================================
In this directory are models and runners for NLP classification experiments. Below is a high-level description of these files. Examples can be found in the [projects/nlp/](https://github.com/numenta/nupic.research/tree/master/projects/nlp) directory. Note the use of the [Cortical.io API](http://api.cortical.io/) for encodings, which we query via cortipy(https://github.com/numenta/cortipy); there is also a [Python SDK by Cortical.io](https://github.com/cortical-io/python-client-sdk).

* classification_model.py: ClassificationModel
	- Specific classification models are subclassed from this base.
    - This base class abstracts the various classification operations used in our NLP experiments. This way the experiment runners (described below) are mostly independent of the specific classification scheme used.

* classify_htm.py: ClassificationModelHTM
    - The network is specified by a network configuration JSON; see projects/nlp/data/network_configs/, e.g.:
        - tp_knn.json: specifies a LanguageSensor --> TP --> KNN HTM network. This model will use a Cortical.io encoder in the LanguageSensor region to encode each word in a sequence, which is then fed to a temporal pooler region, and finally classified in a kNN.
    - HTM networks are instantiated via the [classification network factory](https://github.com/numenta/nupic.research/blob/master/htmresearch/frameworks/classification/classification_network.py).
    - HTM network models will use specialized subclasses of Runner for experiments (described below).

* classify_fingerprint.py: ClassificationModelFingerprint
    - Encode words by querying the Cortical.io API.
    - Includes two types of models (specified by the `fingerprintType` enum) that differ in their encoding logic:
        1. CioWordFingerprint: uses the term/ endpoint of the Cortical.io API to encode each word in the sample text, and then we unionize and sparsify (max 20% sparsity) the encodings to one representative fingerprint. 
        2. CioDocumentFingerprint: uses the text/ endpoint of the Cortical.io API to encode the full sample text into one fingerprint.
    - Feeds the encodings to a kNN classifier.

* classify_keywords.py: ClassificationKeywords
    - Encodes each word with a random SDR.
    - Each word is classified in a kNN. When testing a data sample, the resulting classification is the category most frequent of the inferred words.

* classify_endpoint.py: ClassificationModelEndpoint uses the classification/ endpoint of the Cortical.io API.


* runner.py: Runner
    - Methods for running the NLP classification experiments (see projects/nlp/).
    
* htm_runner.py: HTMRunner
    - Subclass of Runner that implements methods for HTM network models.

