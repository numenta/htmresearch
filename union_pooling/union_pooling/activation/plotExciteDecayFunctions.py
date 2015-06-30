
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------


from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy

"""
This script plots different activation and decay functions and saves the 
resulting figures to a pdf document "excitation_decay_functions.pdf"
"""
with PdfPages('excitation_decay_functions.pdf') as pdf:
	
	plt.figure()
	from union_pooling.activation.excite_functions.logistic_excite_function import (
	  LogisticExciteFunction)
	self = LogisticExciteFunction()
	self.plot()
	plt.title('Sigmoid Activation Function')
	pdf.savefig()
	plt.close()

	# test decay function
	from union_pooling.activation.decay_functions.exponential_decay_function import (
	  ExponentialDecayFunction)

	plt.figure()
	self = ExponentialDecayFunction(10.0)
	self.plot()
	plt.title('Exponential Decay Function')
	pdf.savefig()
	plt.close()

	from union_pooling.activation.decay_functions.logistic_decay_function import (
	  LogisticDecayFunction)

	plt.figure()
	self = LogisticDecayFunction(10.0)
	self.plot()
	plt.title('Sigmoid Decay Function')
	pdf.savefig()
	plt.close()
