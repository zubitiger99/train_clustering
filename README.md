# **🛍️ Mall Customer Segmentation \- K-Means vs DBSCAN**

Interactive dashboard comparing K-Means and DBSCAN clustering algorithms on the Mall Customer Segmentation dataset from Kaggle.

## **🎯 Project Overview**

This project demonstrates:

* K-Means clustering with optimal K selection using Elbow method and Silhouette analysis  
* DBSCAN clustering with parameter optimization  
* Interactive comparison dashboard built with Streamlit  
* Comprehensive metrics and visualizations

## **📊 Dataset**

**Source:** [Kaggle \- Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

**Features:**

* CustomerID  
* Gender  
* Age  
* Annual Income (k$)  
* Spending Score (1-100)

**Clustering Features Used:** Annual Income & Spending Score

## **🚀 Quick Start**

### **Step 1: Download Dataset**

1. Go to [Kaggle Dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)  
2. Download `Mall_Customers.csv`  
3. Place it in the project directory

### **Step 2: Train Models & Generate .pkl File**

\# Install dependencies  
pip install pandas numpy scikit-learn

\# Run the training script  
python train\_clustering.py

This will generate:

* `clustering_models.pkl` \- Contains trained models and data  
* `clustered_data.csv` \- Dataset with cluster labels

### **Step 3: Run Streamlit App Locally**

\# Install streamlit and other dependencies  
pip install \-r requirements.txt

\# Run the app  
streamlit run app.py

The app will open in your browser at `http://localhost:8501`

## **📁 Project Structure**

clustering-project/  
├── train\_clustering.py      \# Training script (generates .pkl file)  
├── app.py                    \# Streamlit dashboard  
├── requirements.txt          \# Python dependencies  
├── README.md                 \# This file  
├── Mall\_Customers.csv        \# Dataset (download from Kaggle)  
├── clustering\_models.pkl     \# Trained models (generated)  
└── clustered\_data.csv        \# Results (generated)

## **🌐 Deploy to Streamlit Cloud**

### **Step 1: Create GitHub Repository**

1. Go to [GitHub](https://github.com/) and create a new repository  
2. Name it: `customer-segmentation-clustering`  
3. Initialize with README

### **Step 2: Upload Files to GitHub**

\# Initialize git in your project folder  
git init

\# Add files  
git add app.py requirements.txt README.md clustering\_models.pkl

\# Commit  
git commit \-m "Initial commit \- Customer segmentation dashboard"

\# Connect to GitHub  
git remote add origin https://github.com/YOUR\_USERNAME/customer-segmentation-clustering.git

\# Push  
git branch \-M main  
git push \-u origin main

**Important:** Make sure `clustering_models.pkl` is uploaded to GitHub\!

### **Step 3: Deploy on Streamlit Cloud**

1. Go to [Streamlit Cloud](https://share.streamlit.io/)  
2. Sign in with GitHub  
3. Click "New app"  
4. Select your repository: `customer-segmentation-clustering`  
5. Set main file path: `app.py`  
6. Click "Deploy"

Your app will be live at: `https://YOUR_APP_NAME.streamlit.app`

## **📊 Features**

### **📈 Overview Page**

* Dataset statistics and distribution  
* Age, income, spending, and gender visualizations  
* Interactive scatter plot

### **🎯 K-Means Analysis**

* Elbow method visualization  
* Silhouette score analysis  
* Cluster profiles and business insights  
* Interactive visualization with centroids

### **🔍 DBSCAN Analysis**

* Automatic cluster detection  
* Noise point identification  
* Parameter optimization results  
* Cluster profiles

### **⚖️ Comparison**

* Side-by-side metrics comparison  
* Visual comparison of both algorithms  
* Detailed comparison table  
* Algorithm recommendation

### **🎮 Interactive Clustering**

* Adjust K for K-Means in real-time  
* Modify DBSCAN parameters (eps, min\_samples)  
* See immediate results

## **📈 Results**

### **K-Means (K=5)**

* **Silhouette Score:** \~0.55  
* **Clusters:** 5 distinct customer segments  
* **Best for:** Well-defined, spherical clusters

### **DBSCAN**

* **Clusters Found:** 4-5 (depending on parameters)  
* **Noise Detection:** Yes (\~5-10% of data)  
* **Best for:** Arbitrary shapes, outlier detection

## **🛠️ Technologies Used**

* **Python 3.8+**  
* **Streamlit** \- Web dashboard  
* **scikit-learn** \- Clustering algorithms  
* **Plotly** \- Interactive visualizations  
* **Pandas & NumPy** \- Data manipulation  
* **Pickle** \- Model serialization

## **📝 Model Training Details**

### **K-Means**

1. Feature scaling using StandardScaler  
2. Elbow method for optimal K (range 2-10)  
3. Silhouette analysis for validation  
4. Final model with K=5

### **DBSCAN**

1. K-distance graph for eps selection  
2. Grid search over eps (0.3-0.7) and min\_samples (4-7)  
3. Best parameters selected based on silhouette score  
4. Noise point detection enabled

## **🎯 Business Applications**

**Identified Customer Segments:**

1. **Careful Spenders** \- Budget-conscious, need discounts  
2. **High Rollers** \- Premium customers, VIP treatment  
3. **Affluent Savers** \- High income but low spending, upselling opportunity  
4. **Average Customers** \- Standard marketing approach  
5. **Aspirational Buyers** \- Low income but high spending, financing options

## **🤝 Contributing**

Feel free to fork this repository and submit pull requests\!

## **📄 License**

This project is open source and available under the MIT License.

## **👨‍💻 Author**

Created with ❤️ for learning and demonstration purposes

## **🔗 Links**

* **Live Demo:** \[Your Streamlit App URL\]  
* **GitHub:** \[Your GitHub Repository URL\]  
* **Dataset:** [Kaggle Link](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)  
* **LinkedIn:** \[Your LinkedIn Profile\]

---

⭐ If you found this helpful, please star the repository\!

