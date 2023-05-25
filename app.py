# Importar las bibliotecas necesarias
from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage

app = Flask(__name__)

# Ruta de inicio
@app.route('/')
def index():
    return render_template('clustering.html')

# Ruta para realizar el clustering
@app.route('/clustering', methods=['POST'])
def clustering():
    try:
        # Obtener el archivo CSV enviado desde el formulario
        csv_file = request.files['csv_file']
        df = pd.read_csv(csv_file)

        # Eliminar las primeras filas no numéricas del DataFrame
        df = df.apply(pd.to_numeric, errors='coerce').dropna()

        if df.shape[0] < 1:
            return "Error: No se encontraron datos numéricos después de la eliminación de filas no numéricas."

        # Preprocesamiento de los datos
        estandarizar = StandardScaler()
        MEstandarizada = estandarizar.fit_transform(df)

        # Clustering jerárquico
        MJerarquico = AgglomerativeClustering(n_clusters=4, linkage='complete', affinity='euclidean')
        MJerarquico_labels = MJerarquico.fit_predict(MEstandarizada)

        # Agregar las etiquetas de clúster al DataFrame
        df['clusterH'] = MJerarquico_labels

        # Calcular los centroides de los clústeres
        CentroidesH = df.groupby(['clusterH']).mean()

        # Crear el gráfico de dispersión
        plt.figure(figsize=(8, 6))
        plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df['clusterH'], cmap='viridis')
        plt.xlabel(df.columns[0])
        plt.ylabel(df.columns[1])
        plt.title('Cluster Scatter Plot')
        plt.colorbar(label='Cluster')
        scatter_plot_path = 'static/scatter_plot.png'
        plt.savefig(scatter_plot_path)
        plt.close()

        # Crear el dendrograma
        Z = linkage(MEstandarizada, method='complete', metric='euclidean')
        plt.figure(figsize=(12, 8))
        dendrogram(Z)
        plt.xlabel('Samples')
        plt.ylabel('Distance')
        plt.title('Dendrogram')
        dendrogram_path = 'static/dendrogram.png'
        plt.savefig(dendrogram_path)
        plt.close()

        # Crear el mapa de calor
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap')
        heatmap_path = 'static/heatmap.png'
        plt.savefig(heatmap_path)
        plt.close()

        # Crear el pairplot
        sns.set(style="ticks")
        pairplot = sns.pairplot(df, hue='clusterH')
        pairplot_path = 'static/pairplot.png'
        pairplot.savefig(pairplot_path)
        plt.close()

        # Pasar los resultados a la plantilla HTML
        return render_template('clustering_results.html', scatter_plot_path=scatter_plot_path, dendrogram_path=dendrogram_path, heatmap_path=heatmap_path, pairplot_path=pairplot_path, centroids=CentroidesH.to_html())

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(host="localhost", port=int("5000"))


