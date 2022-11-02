#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Ilya Ryzhkov / Katarzyna Węsierska
# Created Date: oct 2022
# version = '0.1'
# Algorytm FL oblicza ryzyko nadania dostępu do pomieszczenia na
# podstawie wartości odczytanych z czujników temperatury,
# wilgotności, i wysokości obiektu
# Required: numpy, scikit-fuzzy
# ---------------------------------------------------------------------------
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Nowe Antecedent/Consequent obiekty
humidity = ctrl.Antecedent(np.arange(0, 1, 0.01), 'humidity')
tall = ctrl.Antecedent(np.arange(1, 250, 1), 'tall')
temp = ctrl.Antecedent(np.arange(35, 42, 0.1), 'temp')
risk = ctrl.Consequent(np.arange(0, 1, 0.01), 'risk')

# Mapujemy zakresy liczb
humidity.automf(5, variable_type='quant')
tall.automf(7, variable_type='quant')
temp.automf(3, variable_type='quant')
risk.automf(3, variable_type='quant')

risk['low'] = fuzz.trimf(risk.universe, [0, 0, 0.33])
risk['medium'] = fuzz.trimf(risk.universe, [0, 0.33, 0.5])
risk['high'] = fuzz.trimf(risk.universe, [0.5, 1, 1])

rules = [
    ctrl.Rule(tall['high'] & temp['low'] & humidity['low'], risk['low']),
    ctrl.Rule(tall['higher'] & temp['low'] & humidity['low'], risk['low']),
    ctrl.Rule(tall['highest'] & temp['low'] & humidity['low'], risk['low']),
    ctrl.Rule(tall['high'] & humidity['average'] & temp['low'],
              risk['average']),
    ctrl.Rule(tall['higher'] & humidity['average'] & temp['low'],
              risk['average']),
    ctrl.Rule(tall['highest'] & humidity['average'] & temp['low'],
              risk['average']),
    ctrl.Rule(tall['low'] | tall['average'] | tall['lower'] | tall['lowest'],
              risk['high']),
    ctrl.Rule(humidity['high'] | temp['high'], risk['high'])
]

access_ctrl = ctrl.ControlSystem(rules)
access = ctrl.ControlSystemSimulation(access_ctrl)

access.input['humidity'] = 0.05
access.input['tall'] = 170
access.input['temp'] = 36.6
access.compute()
print('5% 170cm 36.6 C = ' + str(access.output['risk']))
# Wynik: 0.14895755305867667

access.input['humidity'] = 0.3
access.input['tall'] = 170
access.input['temp'] = 36.6
access.compute()
print('30% 170cm 36.6 C = ' + str(access.output['risk']))
# Wynik: 0.3776572454105089

access.input['humidity'] = 0.7
access.input['tall'] = 170
access.input['temp'] = 36.6
access.compute()
print('70% 170cm 36.6 C = ' + str(access.output['risk']))
# Wynik: 0.6809200791795371

access.input['humidity'] = 0.4
access.input['tall'] = 170
access.input['temp'] = 40.0
access.compute()
print('40% 170cm 40 C = ' + str(access.output['risk']))
# Wynik: 0.7955923250321608

access.input['humidity'] = 0.1
access.input['tall'] = 100
access.input['temp'] = 36.6
access.compute()
# Wynik: 0.8093567967688188

print('10% 100cm 36.6 C = ' + str(access.output['risk']))
access.input['humidity'] = 0.7
access.input['tall'] = 100
access.input['temp'] = 36.6
access.compute()
print('70% 100cm 36.6 C = ' + str(access.output['risk']))
# Wynik: 0.8232768179233729

access.input['humidity'] = 0.5
access.input['tall'] = 170
access.input['temp'] = 36.6
access.compute()
print('50% 170cm 36.6 C = ' + str(access.output['risk']))
# Wynik: 0.49506390267089445

access.input['humidity'] = 0.1
access.input['tall'] = 250
access.input['temp'] = 36.6
access.compute()
print('10% 250cm 36.6 C = ' + str(access.output['risk']))
# Wynik: 0.13447960618846697
