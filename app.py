# Importar las bibliotecas necesarias
from flask import Flask, render_template
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
        # Nombre del archivo CSV
        filename = "sample_data.csv"

        # Leer el archivo CSV y realizar el clustering jerárquico
        df = pd.read_csv(filename)
        rango = df.columns.tolist()[2:]  # Excluir las dos primeras columnas
        BCancer = df[rango]

        # Preprocesamiento de los datos
        estandarizar = StandardScaler()
        MEstandarizada = estandarizar.fit_transform(BCancer)

        # Clustering jerárquico
        MJerarquico = AgglomerativeClustering(n_clusters=4, linkage='complete', affinity='euclidean')
        MJerarquico_labels = MJerarquico.fit_predict(MEstandarizada)

        # Agregar las etiquetas de clúster al DataFrame
        BCancer['clusterH'] = MJerarquico_labels

        # Calcular los centroides de los clústeres
        CentroidesH = BCancer.groupby(['clusterH'])[['Texture', 'Area', 'Smoothness', 'Compactness', 'Symmetry', 'FractalDimension']].mean()

        # Crear el dendrograma
        Z = linkage(MEstandarizada, method='complete', metric='euclidean')
        plt.figure(figsize=(12, 8))
        dendrogram(Z)
        plt.xlabel('Samples')
        plt.ylabel('Distance')
        plt.title('Dendrogram')
        plt.savefig('static/dendrogram.png')
        plt.close()

        # Crear el mapa de calor
        plt.figure(figsize=(10, 8))
        sns.heatmap(BCancer.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap')
        plt.savefig('static/heatmap.png')
        plt.close()

        # Crear el pairplot
        sns.set(style="ticks")
        sns.pairplot(BCancer, hue='clusterH')
        plt.savefig('static/pairplot.png')
        plt.close()


        # Pasar los resultados a la plantilla HTML
        return render_template('clustering_results.html', data=BCancer.to_html(), centroids=CentroidesH.to_html(), scatter_poth='static/scatter_plot', heatmap_path='static/heatmap.png', dendrogram_path='static/dendrogram.png', pairplot_path='static/pairplot.png')

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(host="localhost", port=int("5000"))

