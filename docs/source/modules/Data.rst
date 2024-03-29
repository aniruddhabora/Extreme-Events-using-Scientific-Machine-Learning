DATA for E3SM hybrid modelling
==============================

The aim of this project is to develop coarse-scale and data-informed, hybrid methods that produce accurate statistical predictions of climate models. Contemporary climate models run with the resolution needed to capture meteorological events critical to informing decisions for policy makers are computationally too expensive with current computers to capture long timescales.  Affordable coarse-scale climate models allow for long timescales to be simulated, but introduce biases owing to the simplified physical process representations at such grid spacings.  Historically, decadal scale predictability of catastrophic climate events has relied on the blending of large-scale environments from coarse-scale climate models with high resolution, regionally-confined models.  Unfortunately, these methods fail to capture feedback processes between the spatial scales.  Hybrid methods hold promise for a new path forward.

The goal of the presented hybrid methodology is to bridge the gap described above. Specifically, we use short-time climate simulations from highly accurate models, in order to machine-learn a correction that is applied to a coarse-scale model. The necessary corrective tendencies are first estimated from a coarse-scale hindcast simulation which is linearly nudged towards a solution derived from a more accurate model. The machine learning (ML) model is trained to predict the nudging tendencies using only the state of
the coarse-scale model as inputs.  The resulting ML model can then be used in a forecast to apply a corrective tendency to the prognostic state of the atmospheric model at each time step in order to reduce atmospheric model error growth and keep the model evolution on a more realistic manifold.


Climate Model
-------------


To test our proposed ML method, we use the E3SM Atmosphere Model (EAM). EAM uses a Spectral Element Dynamical Core coupled to a sophisticated physics parameterizations package and an interactive land surface model (`Rasch et al., 2019 <https://doi.org/10.1029/2019MS001629>`_, `Xie et al., 2018 <https://doi.org/10.1029/2018MS001350>`_). It is the atmospheric component of the coupled Earth system model developed by the U.S. Department of Energy (`Golaz et al., 2019 <https://doi.org/10.1029/2018MS001603>`_). The standard version of EAM uses a 1-degree (100~km) horizontal grid. There are 72 layers in the vertical, extending from the Earth’s surface to about 0.1 hPa (64 km).  The key subgrid-scale physical processes considered in EAM include deep convection (`Zhang and McFarlane, 1995 <https://doi.org/10.1080/07055900.1995.9649539>`_), turbulence and shallow convection (`Golaz et al., 2002 <https://doi.org/10.1175/1520-0469(2002)059>`_, `Larson et al., 2002 <https://doi.org/10.1175/1520-0469(2002)059>`_), cloud microphysics (`Morrison and Gettelman, 2008 <https://doi.org/10.1175/2008JCLI2105.1>`_, `Gettelman et al., 2015 <https://journals.ametsoc.org/view/journals/clim/28/3/jcli-d-14-00103.1.xml>`_, `Wang et al., 2014 <https://doi.org/10.5194/acp-14-10411-2014>`_), aerosol life cycle (`Liu et al., 2016 <https://doi.org/10.5194/gmd-9-505-2016>`_, `Wang et al., 2020 <https://doi.org/10.1029/2019MS001851>`_), and radiation (`Iacono et al., 2008 <https://doi.org/10.1029/2019MS001851>`_, `Mlawer et al., 1997 <https://doi.org/10.1029/97JD00237>`_).

In our project, EAM is the target model to be improved with the ML-learn correction.  Sea surface temperatures and sea ice extent are prescribed with the observational data from the National Oceanic and Atmospheric Administration (NOAA) Optimum Interpolation (OI) analysis (`Reynolds et al., 2002 <https://journals.ametsoc.org/view/journals/clim/15/13/1520-0442_2002_015_1609_aiisas_2.0.co_2.xml>`_)  The external forcings, including volcanic 
aerosols, solar variability, concentrations of greenhouse gases, and anthropogenic emissions of aerosols and their precursors, were prescribed following the World Climate Research Programme (WCRP) Coupled Model Intercomparison Project Phase 6 (CMIP6) (`Eyring et al., 2016 <https://doi.org/10.5194/gmd-9-1937-2016>`_, `Hoesly et al., 2018 <https://doi.org/10.5194/gmd-11-369-2018>`_, `Feng et al., 2020 <https://doi.org/10.5194/gmd-13-461-2020>`_). 

The EAM model code for the development of ML model can be found [`here <https://github.com/zhangshixuan1987/E3SM/tree/EAM.0_for_darpa>`_]. 



Nudging Approach 
----------------

The nudging approach is employed to estimate the biases in EAM's model state including temperature (T), humidity (Q), zonal wind (U) and meridional wind (V) for the ML training. Here, nudging constrains the model solution of :math:`\boldsymbol{X}_{m}` at every grid point toward the reference state of :math:`\boldsymbol{X}_{\boldsymbol{r}}` by adding a linear relaxation term to the EAM model equation:   

.. math::
    \begin{eqnarray} \label{eqn:eam_nudging}
    \dfrac{\partial \boldsymbol{X_m}}{\partial t} = 
        \underbrace {\boldsymbol{D} \left(\boldsymbol{X_m} \right)}_{dynamics} 
        +  \underbrace {\boldsymbol{P} \left(\boldsymbol{X_m} \right)}_{physics} 
        + \boldsymbol{\dot{R}} 
    \end{eqnarray}

.. math::
    \begin{eqnarray}  \label{eqn:ndg_tend}
    \boldsymbol{\dot{R}} = - \underbrace { \dfrac{ \boldsymbol{X_m} - \boldsymbol{X_r}}{\tau}}_{nudging} 
    \end{eqnarray}

where :math:`\boldsymbol{X}_{m}` and :math:`\boldsymbol{X}_{\boldsymbol{r}}`  refer to the state variables of U, V, T, Q from the EAM predictions and the reference data sets, respectively. The first term on the right-hand side of of the 1st equation represents the effects of large-scale dynamics (e.g. large-scale advection etc.). The second term denotes the parameterized effects of physical processes such as clouds and convection that operate at scales smaller than the model grid and affect the overall dynamics of the system. The third term :math:`\dot{\boldsymbol{R}}` is a nudging tendency term that acts as an error correction for :math:`\boldsymbol{X}_{\boldsymbol{m}}`, calculated as the difference between :math:`\boldsymbol{X}_{\boldsymbol{m}}` and :math:`\boldsymbol{X}_{\boldsymbol{r}}`, scaled by the relaxation time scale :math:`\tau` in the 2nd equation.

Pink boxes in Figure 1 illustrate where the nudging-related calculations occur in the default EAM. In a nudged simulation,  after the resolved dynamics (see blue box in figure) has been calculated,  a nudging tendency term in the form of Eq 2 is calculated for each nudged variable with  :math:`\boldsymbol{X}_{m}` being the value of $X$ after the dynamical core. After the entire physics parameterization suite has been calculated, the sum of the parameterization-induced tendencies and the nudging tendencies are passed to the physics-dynamics coupling interface.


.. figure:: Data_figs/e3sm_nudging_flow.png
  :width: 700
  :align: center
  :alt: Alternative text

  Figure 1: Flowcharts showing the sequence of dynamics and physics calculations within one time step in an EAM simulation. Pink boxes indicate where the nudging-related calculations occur. The calculation of nudging tendency using Eq. (2) occurs before the radiation parameterization.


Nudged training simulation with EAM
-----------------------------------

In Phase 1, the ML training data are constructed following a "nudge-to-observations" approach described in `Watt-Meyer et. al. (2021) <https://doi.org/10.1029/2021GL092555>`_. In the "nudge-to-observations"  approach employed by this project, the observations (i.e. reference data sets) are taken from the ERA5 reanalysis developed by the European Centre for Medium-Range Weather Forecasts (ECMWF) (`Hersbach et al., 2020 <https://doi.org/10.1002/qj.3803>`_). The raw ERA5 reanalysis data are produced on a :math:`0.25^{o}` horizontal grid over the globe, which are spatially remapped to the cubed-sphere grid and the 72 model layers used by EAM, following the method used in the Community Earth System Model Version 2 [`CESM2 <https://ncar.github.io/CAM/doc/build/html/users_guide/physics-modifications-via-the-namelist.html#nudging>`_]. Topographical differences between EAM and the reanalysis data are taken into account during the vertical interpolation. 

Figure 2a shows the distribution of monthly mean zonal averaged temperature differences between the EAM free-running simulations (CLIM) and ERA5 reanalysis (reference) in January 2010. Most model layers in the Tropics and mid-latitudes exhibit a cold temperature bias. In these regions, the positive temperature nudging tendencies in the nudged simulation act to correct the cold biases (Fig.2b).  Generally the time mean nudging tendency removes the systematic "background error" found in the EAM free-running simulations. However, the nudging may not always help to reduce the systematic errors. For example, nudging both wind and temperature can produce a positive tendency of temperature in the northern hemisphere high-latitude (Fig.2b), where the free-running simulations exhibit warm temperature biases, as shown in Fig.2a, suggesting a role of positive feedback that amplifies the upper level temperature biases in the free-running simulations. Using a nudging strategy that constrains humidity in addition to wind and temperature produces rather different nudging tendencies (Fig.2c), revealing the complex relationships between the nudging corrections and the state variables through the nonlinear governing equation (see equation on top of the page). Therefore, we design different nudging strategies to provide an ensemble of nudged simulations with different nudging tendencies and state variables for the ML training.

.. figure:: Data_figs/mean_bias.png
  :width: 800
  :align: center
  :alt: Alternative text
  
  Figure 2 (a) monthly mean zonally averaged temperature differences (ΔT, unit: K) in January 2010 between ERA5 and EAM's free-running simulation (CLIM in Table 1), (b-c) monthly mean nudging tendencies of temperature (T tend, unit K s−1) from the simulation by nudging EAM towards ERA5 reanalysis. The wind and temperature fields were nudged in the simulation (NDG_UVT in Tabel 1) for panel (b), while the wind, temperature and humidity were nudged in the simulation (NDG_UVTQ in Table 1) for panel (c). The y-axis of each panel shows the approximated pressure for the model levels in EAM.

Three groups of training data are generated in phase 1 (Table~\ref{tabtrainning_exp}). The first group consists of the reference solution for U, V, T, Q that are derived from ERA5 reanalysis. The data are interpolated to the same grid and vertical levels for E3SM. The second group is a free-running baseline simulation referred to as CLIM. The before-radiation values of U, V, T, Q were archived to represent the baseline solution from the EAM-LR. The third group of simulations was nudged toward ERA5 reanalysis to derive the corrective tendencies of U, V, T, Q for ML training. The three pairs of  simulation are conducted to construct an ensemble of training data sets by applying nudging:

- to the horizontal winds with :math:`\tau` = 6  (labeled "NDG\_UV")
- to both winds and temperature  with :math:`\tau` = 6 (labeled "NDG\_UVT")
- to winds, temperature, and humidity :math:`\tau` = 6 (labeled "NDG\_UVTQ") 


.. figure:: Data_figs/table_1.png
  :width: 800
  :align: center
  :alt: Alternative text

  Table 1 List of reference data and EAM-LR simulations for machine learning. Note nudging is applied at every model physics time step (0.5-hr) for EAM.



All EAM simulations were conducted for 11-years from 2007 to 2017. The first year is for model spin-up and the remaining 10-years are used to construct the input data for ML training. Table 2 presents the list of the input variables for ML training. The  3-D model state (U, V, T, Q)  variables are the instantaneous model output, while the nudging tendencies are averaged values during a 3-hr period for each time sample. The data are available at this [`link <https://portal.nersc.gov/cfs/e3sm/zhan391/darpa_temporary_data_share/SE_PG2/>`_]

.. figure:: Data_figs/table_2.png
  :width: 600
  :align: center
  :alt: Alternative text

  Table 2 Description of notation. Notes: the (x, y, z, t) is corresponding to the (latitude, logitude, levels, time) dimension in the EAM model output. Each notation contains the four state variables (i.e. U, V, T, Q) that are interested in this projec


