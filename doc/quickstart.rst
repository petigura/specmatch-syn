.. _quickstart:

Quickstart Tutorial
===============

Shifting Spectrum onto Model Wavelength Scale
---------------------------------------------

.. code-block:: python

   lib = smsyn.library.read_hdf(<path to coelho models>,wavlim=[4000,7000])
   hduL = fits.open(<path to spectrum>)
   wav = hduL[2].data
   uflux = hduL[1].data
   flux = hduL[0].data
   flux /= np.percentile(flux,95,axis=1)[:,np.newaxis]
   ref_wav = np.logspace(np.log10(np.min(wav)),np.log10(np.max(wav)),64000)
   ref_flux  = lib.synth(ref_wav,5700,4.5,0.0,1.0,0,)
   flux_shift, uflux_shift = smsyn.inst.hires.shift.shift(wav, flux, uflux, ref_wav, ref_flux)
   spec = smsyn.io.spectrum.Spectrum(ref_wav, flux_shift, uflux_shift, header=dct(name=name,obs=obs))
   spec.to_fits('output/{}_{}.sm.fits'.format(name,obs)) # Save to output

Run SpecMatch Algorithm
-----------------------

.. code-block:: python 

    from smsyn.inst.hires.pipeline import Pipeline
    outfile = 'output/{}_{}.sm.fits'.format(name,obs)
    libfile = '/Users/petigura/Dropbox/fulton_petigura/coelho05_2.hdf'
    pipe = Pipeline(outfile, libfile)
    pipe.grid_search(debug=False)
    pipe.lincomb()





