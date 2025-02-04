import pandas as pd 
import os 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print(os.getcwd())

#LEER LOS DATOS
#________________________________________
features_df=pd.read_csv("Features_data set.csv")
sales_df = pd.read_csv("sales_data-set.csv")
stores_df=pd.read_csv("stores_data-set.csv")

'''
print("Features Data:")
print(features_df.head())
print("Features Data:")
print(sales_df.head())
print("Features Data:")
print(stores_df.head())
'''

print( "features Data shape:", features_df.shape  )
print(  "Sales Data shape:", sales_df.shape)
print( "stores_df:", stores_df.shape)

print("Features:", features_df.columns.tolist())
print("Sales:", sales_df.columns.tolist())
print("Stores:", stores_df.columns.tolist())


# Agrupar ventas semanales por tienda y fecha
sales_grouped = sales_df.groupby(["Store", "Date"])["Weekly_Sales"].sum().reset_index()

# Unimos con features_df
merged_df = sales_grouped.merge(features_df, on=["Store", "Date"], how="left")

print(merged_df.head())

print(sales_grouped.describe())  # Estadísticas generales
print(sales_grouped.sort_values(by="Weekly_Sales", ascending=False).head(10))  # Las 10 tiendas con más ventas

##LIMPIEZA DE DATOS Y VALORES NULOS#----------------
print("Buscar valores nulos por columna")
print(merged_df.isnull().sum())

# Rellenar valores nulos en MarkDown con 0
merged_df[['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']] = merged_df[['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']].fillna(0)

# Confirmar que ya no hay valores nulos
print(merged_df.isnull().sum())

print(merged_df["Date"].head(70))

##CONVERTIR FECHA A Datetime_----------------------------

# Intentar convertir 'Date' a datetime, pero marcando errores como NaT
merged_df["Date_Converted"] = pd.to_datetime(merged_df["Date"], errors='coerce')

# Filtrar y mostrar filas donde la conversión falló (se convirtieron en NaT)
invalid_dates = merged_df[merged_df["Date_Converted"].isna()]["Date"].unique()
print("Fechas problemáticas:", invalid_dates)


# 1. Limpiar espacios en blanco en la columna Date
merged_df["Date"] = merged_df["Date"].str.strip()

# 2. Asegurar que los separadores sean "/"
merged_df["Date"] = merged_df["Date"].str.replace("-", "/")

# 3. Convertir a datetime con dayfirst=True
merged_df["Date"] = pd.to_datetime(merged_df["Date"], errors='coerce', dayfirst=True)

# 4. Revisar cuántas fechas quedaron como NaT (no convertidas)
print("Fechas inválidas después de la conversión:", merged_df["Date"].isnull().sum())


# Convertir 'Date' a formato de fecha
merged_df['Date'] = pd.to_datetime(merged_df['Date'])
print(merged_df.head())
print(merged_df.dtypes)




###ANALISIS DE DATOS_____________________________________

import matplotlib.pyplot as plt

# Graficar la tendencia de ventas en el tiempo
ventas_tiempo= merged_df.groupby("Date")["Weekly_Sales"].sum()
plt.figure(1)
plt.plot(ventas_tiempo, label="Ventas Totales", color="blue")
plt.xlabel("Fecha")
plt.ylabel("Ventas (USD)")
plt.title('Tendencia de Ventas Semanales')
plt.legend()
plt.grid(True)

# Comparar ventas en semanas festivas vs no festivas
ventas_festivos = merged_df.groupby("IsHoliday")["Weekly_Sales"].mean()

plt.figure(figsize=(7, 5))
plt.bar(["No Festivo", "Festivo"], ventas_festivos, color=["gray", "red"])
plt.xlabel("Día Festivo")  #
plt.ylabel("Ventas Promedio ($USD)")
plt.title("Relación de Días Festivos con las Ventas")
plt.grid(axis="y")



#Precio de combustible contra ventas.
plt.figure(3)
plt.scatter(merged_df["Fuel_Price"], merged_df["Weekly_Sales"], alpha=0.5, color="green")
plt.title("Relación entre Precio del Combustible y Ventas")
plt.xlabel("Precio del Combustible ($)")
plt.ylabel("Ventas Semanales ($USD)")
plt.grid(True)


##Ventas agrupadas por mes

merged_df["Month"] = merged_df["Date"].dt.month
ventas_por_mes=merged_df.groupby("Month")["Weekly_Sales"].mean()
plt.figure(4)
ventas_por_mes.plot(kind="bar", color="purple")
plt.xlabel("Mes")
plt.ylabel("Ventas Promedio (USD)")
plt.title("Promedio de Ventas por Mes")
plt.xticks(range(12), ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"], rotation=0)

# Unir con la tabla de tiendas para agregar información del tamaño
merged_df = merged_df.merge(stores_df, on="Store", how="left")

# Agrupar por tipo de tienda y calcular promedio de ventas
ventas_por_tienda = merged_df.groupby("Size")["Weekly_Sales"].mean().sort_index()

# Graficar
plt.figure(figsize=(8, 5))
plt.scatter(ventas_por_tienda.index, ventas_por_tienda.values, color="orange")
plt.xlabel("Tamaño de la Tienda")
plt.ylabel("Ventas Promedio ($USD)")
plt.title("Relación entre el Tamaño de la Tienda y Ventas Promedio")
plt.grid(True)



#plt.show()






######## Haremos una regresión lineal simple para predecir las ventas en función del tamaño de la tienda----------------------
#-----------------------------------------------------------------------------------------------------------------------------

X = merged_df[["Size"]]  # Variable independiente (predictora)
y = merged_df["Weekly_Sales"]  # Variable dependiente (objetivo)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
##MODELO
modelo = LinearRegression()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

#Evaluar el modelo
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")  # Error absoluto medio
print(f"MSE: {mse:.2f}")  # Error cuadrático medio
print(f"R²: {r2:.4f}")  # Coeficiente de determinación

plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, color="orange", label="Datos Reales")
plt.plot(X_test, y_pred, color="blue", linewidth=2, label="Regresión Lineal")
plt.xlabel("Tamaño de la Tienda (Size)")
plt.ylabel("Ventas Semanales ($USD)")
plt.title("Regresión Lineal: Tamaño de la Tienda vs Ventas")
plt.legend()
plt.grid(True)


plt.show()


######## Haremos una regresión multiple para ver si se reduce el error absoluto----------------------
#-----------------------------------------------------------------------------------------------------------------------------

X = merged_df[["Size", "Fuel_Price", "CPI", "Unemployment", "IsHoliday"]]
y = merged_df["Weekly_Sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo_multiple = LinearRegression()
modelo_multiple.fit(X_train, y_train)

y_pred_multi = modelo_multiple.predict(X_test)

mae_multi = mean_absolute_error(y_test, y_pred_multi)
mse_multi = mean_squared_error(y_test, y_pred_multi)
r2_multi = r2_score(y_test, y_pred_multi)

print(f"MAE (Regresión Múltiple): {mae_multi:.2f}")
print(f"MSE (Regresión Múltiple): {mse_multi:.2f}")
print(f"R² (Regresión Múltiple): {r2_multi:.4f}")

coef_df = pd.DataFrame(modelo_multiple.coef_, X.columns, columns=["Coeficiente"])
print("\nCoeficientes del modelo:")
print(coef_df)