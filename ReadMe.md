# **Análisis de Ventas en Tiendas Minoristas**

## **1. Descripción del Proyecto**

Este proyecto analiza los factores que afectan las ventas semanales de tiendas minoristas utilizando tres conjuntos de datos: `features_data`, `sales_data` y `stores_data`. El objetivo es identificar tendencias, patrones estacionales y factores relevantes que impactan en las ventas.

## **2. Objetivos**

- Analizar cómo las variables como el tamaño de la tienda, días festivos, precios de combustible y tasas de desempleo afectan las ventas.
- Explorar relaciones y tendencias en los datos mediante visualizaciones.
- Construir un modelo predictivo utilizando regresión lineal y regresión múltiple.

## **3. Descripción de archivos usados para el análisis**

- **`Features_data`**: Información sobre condiciones externas y eventos.
- **`Sales_data`**: Ventas semanales de diferentes tiendas y departamentos.
- **`Stores_data`**: Información sobre el tamaño y tipo de tiendas.

## **4. Análisis Exploratorio**

Analizamos la distribución de algunas variables clave, como la temperatura, el precio del combustible, el índice de precios al consumidor (CPI) y la tasa de desempleo. Esto nos permite observar patrones generales en los datos.

### **Distribución de Variables Clave**



- **Temperatura**: Distribución normal entre 0 y 100.
- **Precio del Combustible**: Presenta varias modas, sugiriendo fluctuaciones en los precios.
- **CPI (Inflación)**: Variaciones significativas en diferentes periodos.
- **Tasa de Desempleo**: Distribuida entre 4% y 14%, con un pico alrededor del 8%.

## **5. Tendencia de Ventas por Semana**

Visualizamos la evolución de las ventas a lo largo del tiempo (2010-2012) y encontramos picos en noviembre y diciembre.



Esto se debe a eventos comerciales como **Black Friday** y la temporada navideña.

## **6. Relación entre Días Festivos y Ventas**

Analizamos si hay un incremento en las ventas durante semanas festivas.



Se observa un ligero incremento en ventas durante días festivos, aunque la diferencia no es muy grande.

## **7. Relación entre Precio del Combustible y Ventas**

Exploramos si hay una correlación entre el precio del combustible y las ventas semanales.



No se observa una relación clara, lo que sugiere que los consumidores de estas tiendas no son altamente sensibles a cambios en el precio del combustible.

## **8. Ventas Promedio por Mes**

Visualizamos cómo varían las ventas a lo largo del año.



Las ventas aumentan significativamente en noviembre y diciembre, lo que refuerza la hipótesis de impacto estacional.

## **9. Regresión Lineal**

Dado que el tamaño de la tienda parece ser un factor determinante, realizamos una regresión lineal para cuantificar su impacto.



Los resultados muestran:

- **MAE (Error Absoluto Medio)**: \$247,899.37\$
- **MSE (Error Cuadrático Medio)**: \$1.96 \times 10^8\$
- **Coeficiente de Determinación (\$R^2\$)**: 0.6689 (\~67%)

Esto indica que el tamaño de la tienda explica **\~67%** de la variabilidad en las ventas.

## **10. Regresión Múltiple**

Agregamos variables adicionales: precio del combustible, CPI, tasa de desempleo y días festivos.

### **Resultados del Modelo**

| Métrica | Regresión Simple     | Regresión Múltiple   |
| ------- | -------------------- | -------------------- |
| MAE     | 247,899.37           | 246,638.26           |
| MSE     | \$1.96 \times 10^8\$ | \$1.02 \times 10^8\$ |
| \$R^2\$ | 0.6689               | 0.6780               |

### **Coeficientes del Modelo**

| Variable               | Coeficiente |
| ---------------------- | ----------- |
| Tamaño de la tienda    | 7.17        |
| Precio del Combustible | -12,634.00  |
| CPI                    | 2,176.89    |
| Desempleo              | -17,773.89  |
| Días Festivos          | 86,617.28   |

### **Conclusiones**

1. **El modelo de regresión múltiple mejora la predicción** respecto al modelo simple, con una ligera reducción del error absoluto y cuadrático medio.
2. **El tamaño de la tienda sigue siendo el factor más relevante**, explicando un 67.8% de la variabilidad en las ventas.
3. **El precio del combustible y la inflación tienen efectos menores**, mientras que el desempleo tiene un impacto negativo en las ventas.
4. **Las semanas festivas aumentan significativamente las ventas**, lo que es esperable en el comercio minorista.

## **11. Recomendaciones**

- Implementar estrategias de marketing para aumentar ventas fuera de la temporada alta.
- Monitorear el desempleo y otros factores macroeconómicos.
- Optimizar la logística en temporadas de alto consumo.

