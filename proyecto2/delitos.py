#%%
import numpy as np
import pandas as pd

#%%

delitos1 = pd.read_csv('./IDM_jun2022.csv', encoding='ISO-8859-1')
delitos2 = pd.read_csv('./Municipal-Delitos-2015-2023_mar2023.csv', encoding='ISO-8859-1')

#%% Limpieza y estandarización de ambas bases de datos
delitos1.columns = ['AÑO', 'INEGI', 'ENTIDAD', 'MUNICIPIO', 'MODALIDAD', 'TIPO', 'SUBTIPO',
               'ENERO', 'FEBRERO', 'MARZO', 'ABRIL', 'MAYO', 'JUNIO', 'JULIO',
               'AGOSTO', 'SEPTIEMBRE', 'OCTUBRE', 'NOVIEMBRE', 'DICIEMBRE']

delitos1["MODO"] = np.nan

delitos1 = delitos1[['AÑO', 'ENTIDAD', 'MUNICIPIO', 'MODALIDAD', 'TIPO', 'SUBTIPO',
           'MODO', 'ENERO', 'FEBRERO', 'MARZO', 'ABRIL', 'MAYO', 'JUNIO', 'JULIO',
           'AGOSTO', 'SEPTIEMBRE', 'OCTUBRE', 'NOVIEMBRE', 'DICIEMBRE']]


delitos2.columns = ['AÑO', 'CLAVE_ENT', 'ENTIDAD', 'CLAVE_MUNICIPIO', 'MUNICIPIO',
               'MODALIDAD', 'TIPO', 'SUBTIPO', 'MODO', 'ENERO', 'FEBRERO',
               'MARZO', 'ABRIL', 'MAYO', 'JUNIO', 'JULIO', 'AGOSTO', 'SEPTIEMBRE',
               'OCTUBRE', 'NOVIEMBRE', 'DICIEMBRE']

delitos2 = delitos2[['AÑO', 'ENTIDAD', 'MUNICIPIO', 'MODALIDAD', 'TIPO', 'SUBTIPO',
           'MODO', 'ENERO', 'FEBRERO', 'MARZO', 'ABRIL', 'MAYO', 'JUNIO', 'JULIO',
           'AGOSTO', 'SEPTIEMBRE', 'OCTUBRE', 'NOVIEMBRE', 'DICIEMBRE']]

#%%
def get_Delitos(MUNICIPIO, municipio):
    datos1 = delitos1[delitos1['MUNICIPIO'] == MUNICIPIO]
    datos2 = delitos2[delitos2['MUNICIPIO'] == municipio]

    delitos = pd.concat([datos1, datos2])

    return delitos


#%%
apaseo = get_Delitos('APASEO EL GRANDE', 'Apaseo el Grande')
chona = get_Delitos('ENCARNACIÓN DE DÍAZ', 'Encarnación de Díaz')
fresnillo = get_Delitos('FRESNILLO', 'Fresnillo')
uruapan = get_Delitos('URUAPAN', 'Uruapan')
sanAntonio = get_Delitos('COYUCA DE CATALÁN', 'Coyuca de Catalán')
heliodoro = get_Delitos('LEONARDO BRAVO', 'Leonardo Bravo')


#%%
todos = pd.DataFrame()
todos = pd.concat([todos, apaseo])
todos = pd.concat([todos, chona])
todos = pd.concat([todos, fresnillo])
todos = pd.concat([todos, uruapan])
todos = pd.concat([todos, sanAntonio])
todos = pd.concat([todos, heliodoro])


#%%
todos


#%%
writer = pd.ExcelWriter('delitos.xlsx', engine='xlsxwriter')
workbook = writer.book

todos.to_excel(writer, sheet_name='All', index=False)
fresnillo.to_excel(writer, sheet_name="Fresnillo", index=False)
uruapan.to_excel(writer, sheet_name="Uruapan", index=False)
sanAntonio.to_excel(writer, sheet_name="Coyuca de Catalán", index=False)
heliodoro.to_excel(writer, sheet_name="Leonardo Bravo", index=False)
apaseo.to_excel(writer, sheet_name="Apaseo el Grande", index=False)
chona.to_excel(writer, sheet_name="Encarnación de Díaz", index=False)

writer.close()
