Non-intrusive LSTM architecture
===============================

This work aims to train a neural network that, given as input the predictions of a free running coarse-scale simulation, denoted as CLIM in this project, :math:`\left( U, V, Q, T \right)^{\text{CLIM}}`, it will produce a modified time-series  :math:`\left( U, V, Q, T \right)^{\text{ML}}` that will have the same statistics as a fine-scale reference simulation  :math:`\left( U, V, Q, T \right)^{\text{ERA5}}`. For this project, reference data correspond to ERA5 reanalysis datasets while free running coarse-scale simulations are generated via the E3SM CLIM model. A diagram of this process is described in~\cref{fig:Methodology_Diagram}. The de-coupling of the data-informed correction process and the initial simulation phase is justified by the fact that the goal is not to make phase corrections at each time-step but retrieve the correct statistics for the current flow parameters. 


.. figure:: images/Methodology_Plot.png
  :width: 600
  :align: center
  :alt: Alternative text

  Figure 1: While testing will be carried out using free running coarse-scale data, appropriate training data need to be determined first. Due to chaotic divergence, free running coarse-scale data will very quickly diverge from their fine-scale conuterpart despite having the same flow parameters and initial conditions. As a result, it is not feasible for a neural network to learn a generalizable mapping directly between :math:`\left( U, V, Q, T \right)^{\text{CLIM}}`` and :math:`\left( U, V, Q, T \right)^{\text{ERA5}}`. To that end, to produce coarse-scale simulations for training, a relaxation term $Q$ is added to the evolution equations of the prognostic variables :math:`\left( U, V, T, Q\right)`. The term $Q$ is called nudging tendency and it corrects the coarse-scale solution based on the fine-scale reference solution. In this study, for a variable :math:`X`, the nudging tendency :math:`Q`` is given by the algebraic term


Model Architecture
------------------

This subsection discusses how recurrent neural networks (RNN) are used for the data-informed mappings previously described. In particular, long short-term memory (LSTM)~\cite{hochreiter1997long} neural networks are employed. Of great interest is the ability of this model to generalize beyond the data seen during training. At first this is investigated in out-of-sample data from the training flow and later further tested on different flow setups. The architecture of the LSTM-based neural-network is shown in~\cref{fig:LSTM_Architecture}. It consists of an input fully connected layer that compresses prognostic variables of a single level to a $600$-valued vector. This layer has a $\tanh$ activation function. The compressed vector is then passed as input to a long short-term memory (LSTM) neural network. The output of the neural network is then passed through an output fully connected neural network to produce the final data-informed corrected predictions. The output layer has a linear activation function.


.. figure:: images/ML_Architecture.png
  :width: 600
  :align: center
  :alt: Alternative text
  LSTM_Architecture


Data Preparation
----------------

LSTM neural networks incorporate (non-Markovian) memory effects into the reduced-order model. This ability stems from Takens embedding theorem~\cite{takens1981detecting}. The theorem states that given delayed embeddings of a limited number of state variables, one can still obtain the attractor of the full system for the observed variables. This approach is known to be capable of improving predictions of reduced-order models~\cite{ vlachas2018data, charalampopoulos2022machine, wan2021}. Hence, it is expected that RNNs can help predict the contribution of unresolved scales. 

When training with nudged data, a main reason for discrepancies during testing is due to different statistical behaviour of the nudged solution with respect to the free-running coarse data. This is a result of discrepancies in the energy spectrum of the nudged solution with respect to the coarse-scale solution. These energy spectra differences lead to different statistical behaviours of testing data $\left( U, V, Q, T \right)^{\text{CLIM}}$ and training data $\left( U, V, Q, T \right)^{\text{Nudged}}$. 
Discrepancies in the training and testing input distributions will lead to the neural network behaving differently in the two schemes~\cite{shalev2014understanding}. These discrepancies cannot be reconciled by simply choosing an appropriate $\tau$ as algebraic nudging adds linear dissipation to the system, thus always changing the energy spectrum of the resulting flow. 
To remedy the energy spectra differences, a new method is developed and employed. The process is called `Reverse Spectral Nudging' with its purpose being to match the energy spectrum of the nudged solution to that of the coarse-scale solution to improve the training process. Hence, while traditional nudging schemes correct the coarse-scale solution with data from the reference solution, the proposed scheme further processes the nudged data by matching its energy spectrum to that of the corresponding free running coarse-scale flow. The corrected nudged data is termed as $\left( U,V,T,Q \right)^{\text{RS-nudge}}$ and defined, for a prognostic variable $X$, as

.. math::
    \begin{gather}\label{eq:R_nudge}
    \begin{split}
        X^{\text{RS-nudge}}\left(x, y t; z=z_0\right) = \sum_{k,l} R_{k,l} \hat{X}_{k,l}^{\text{nudge}}(t;z=z_0) e^{i\left( k x +l y \right)}, ,
    \end{split}
    \end{gather}
    
where ${X}_{k,l}^{\text{nudge}}(t)$ are the spatial Fourier coefficients of $X^{\text{nudge}}$ and   

.. math::
    \begin{gather}\label{eq:Rcoeff}
    \begin{split}
        R_{k,l} = \sqrt{\frac{\mathcal{E}^{\text{CLIM}}_{k,l}}{\mathcal{E}^{\text{nudge}}_{k,l}}}, \quad \text{and} \quad
        \mathcal{E}_{k,l} = \frac{1}{T} \int_0^T \hat{E}_{k,l}(t) \mathrm{d}t=\frac{1}{T} \int_0^T |\hat{X}_{k,l}(t)|^2 \mathrm{d}t.
    \end{split}
    \end{gather}

A depiction of the values of these coefficients can be seen in figure~\cref{fig:Rcoeffs}.

.. figure:: images/Rcoef.png
  :width: 600
  :align: center
  :alt: Alternative text
  :math:`R`-coefficients for each one of the prognostic variables. Values are shown for the surface layer

An important property of this scheme is that the new data have exactly the energy spectrum of the free running coarse simulation, meaning that the training and testing data come from the same distributions. This property improves significantly the accuracy of the resulted ML scheme. The energy spectra of the R-nudged solution indeed coincide with the coarse-scale free running spectra. In addition, the R-nudged data still follow the reference data, allowing for a mapping between :math:`$\left( U,V,T,Q\right)^{\text{R-Nudge}}$` and :math:`$\left( U,V,T,Q \right)^{\text{ERA5}}$`. This process does not require running additional nudged simulations, thus lowering the total cost of the training scheme.