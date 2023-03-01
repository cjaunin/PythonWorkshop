#!/usr/bin/env python3
########################################################################################################################
# EMF-QARM - Python Workshop - Parameters
# Authors: Maxime Borel, Coralie Jaunin
# Creation Date: February 27, 2023
# Revised on: February 27, 2023
########################################################################################################################
# paths
path = {'Main': '/Users/cjaunin2/PycharmProjects/PythonWorkshop'}
path.update({'Code': path.get('Main') + '/Code',
             'Inputs': path.get('Main') + '/Inputs',
             'Outputs': path.get('Main') + '/Outputs'})
########################################################################################################################
# parameters
start_date = '01-01-2015'
end_date = '12-31-2017'
firm_sample_ls = ['AAL', 'NKE', 'T', 'VZ']
