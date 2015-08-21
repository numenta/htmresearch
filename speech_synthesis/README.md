This folder will eventually contain code for speech synthesis using HTMs.

Currently it contains a letter encoder that takes an English letter and maps it to the phonetic alphabet.  This mapping is surjective for all sounds in the American English phonetic alphabet.  It is not injective for English.

This mapping is based on the International Phonetic Alphabet, but currently only represents the English alphabet. There are still many aspects of IPA that we can add to the encoder to better represent the letters.

IPA chart: http://www.internationalphoneticalphabet.org/ipa-sounds/ipa-chart-with-sounds/

Work for the future:
-Encode other alphabets so we cover the entire IPA
-See how the output from temporal memory can be converted to sound
-How can we handle different dialects
