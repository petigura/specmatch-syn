from setuptools import setup, find_packages

setup(
    name="smsyn-package",
    version="2.0b",
    author="Erik Petigura, BJ Fulton",
    packages = ['smsyn', 'smsyn.io', 'smsyn.inst', 'smsyn.plotting'],
    data_files = [
        (
            'smsyn_example_data', 
            [
                'smsyn/data/10700_raad.132.fits' 
            ]
        )
    ]
    )
