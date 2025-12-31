# **🚀 Complete Setup Guide**

Follow these steps exactly to get your clustering dashboard up and running\!

## **✅ Prerequisites**

* Python 3.8 or higher  
* Git installed  
* GitHub account  
* Kaggle account

---

## **📥 Step 1: Download the Dataset**

1. Visit [Kaggle Dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)  
2. Click "Download" (you may need to log in)  
3. Extract `Mall_Customers.csv` from the zip file  
4. Save it to your project folder

---

## **🔧 Step 2: Create Project Folder & Files**

Create a folder structure:

customer-segmentation/  
├── Mall\_Customers.csv          (downloaded from Kaggle)  
├── train\_clustering.py         (copy from artifacts)  
├── app.py                      (copy from artifacts)  
├── requirements.txt            (copy from artifacts)  
└── README.md                   (copy from artifacts)

---

## **🎯 Step 3: Train Models & Generate .pkl File**

Open terminal/command prompt in your project folder:

\# Install required libraries  
pip install pandas numpy scikit-learn

\# Run training script  
python train\_clustering.py

**Expected Output:**

* Console will show clustering progress and results  
* Files created:  
  * ✅ `clustering_models.pkl` (\~50-100 KB)  
  * ✅ `clustered_data.csv`

**Verify:** Check that `clustering_models.pkl` was created\!

---

## **💻 Step 4: Test Locally**

\# Install all dependencies  
pip install \-r requirements.txt

\# Run Streamlit app  
streamlit run app.py

**Expected Result:**

* Browser opens automatically at `http://localhost:8501`  
* Dashboard loads with all 5 pages working  
* No errors in console

**Test all pages:**

* ✅ Overview  
* ✅ K-Means Analysis  
* ✅ DBSCAN Analysis  
* ✅ Comparison  
* ✅ Interactive Clustering

---

## **📦 Step 5: Create GitHub Repository**

### **Option A: Via GitHub Website**

1. Go to [github.com](https://github.com/)  
2. Click "+" → "New repository"  
3. Repository name: `customer-segmentation-clustering`  
4. Description: `K-Means vs DBSCAN clustering on mall customers`  
5. Select: **Public** ✅  
6. ❌ Don't initialize with README (you already have one)  
7. Click "Create repository"

### **Option B: Via Command Line**

\# Initialize git in your project folder  
cd customer-segmentation  
git init

\# Add all files  
git add .

\# Make first commit  
git commit \-m "Initial commit: Customer segmentation dashboard"

\# Create new repo on GitHub (you'll need GitHub CLI)  
\# OR follow Option A and then:

\# Connect to GitHub repository  
git remote add origin https://github.com/YOUR\_USERNAME/customer-segmentation-clustering.git

\# Push to GitHub  
git branch \-M main  
git push \-u origin main

**Important Files to Upload:**

* ✅ `app.py`  
* ✅ `requirements.txt`  
* ✅ `README.md`  
* ✅ `clustering_models.pkl` ⚠️ **CRITICAL \- Don't forget\!**  
* ❌ `Mall_Customers.csv` (optional, already in .pkl)  
* ❌ `train_clustering.py` (optional, for reference)

---

## **🌐 Step 6: Deploy to Streamlit Cloud**

### **6.1 Access Streamlit Cloud**

1. Go to [share.streamlit.io](https://share.streamlit.io/)  
2. Click "Sign in with GitHub"  
3. Authorize Streamlit to access your repositories

### **6.2 Deploy Your App**

1. Click **"New app"** button

2. Fill in the form:

   * **Repository:** `YOUR_USERNAME/customer-segmentation-clustering`  
   * **Branch:** `main`  
   * **Main file path:** `app.py`  
   * **App URL:** Choose a custom name (e.g., `customer-segmentation-demo`)  
3. Click **"Deploy\!"**

### **6.3 Wait for Deployment**

* ⏳ Takes 2-5 minutes  
* You'll see logs in real-time  
* Look for: "Your app is live\! 🎉"

### **6.4 Get Your Live URL**

Your app will be available at:

https://YOUR-APP-NAME.streamlit.app

---

## **✅ Verification Checklist**

After deployment, test your live app:

* \[ \] App loads without errors  
* \[ \] Overview page shows charts  
* \[ \] K-Means analysis displays properly  
* \[ \] DBSCAN analysis works  
* \[ \] Comparison page shows both algorithms  
* \[ \] Interactive clustering responds to slider changes  
* \[ \] No "FileNotFoundError" for .pkl file

---

## **🐛 Troubleshooting**

### **Problem: "FileNotFoundError: clustering\_models.pkl"**

**Solution:**

\# Make sure .pkl file is in your repository  
git add clustering\_models.pkl  
git commit \-m "Add clustering models"  
git push

Then redeploy in Streamlit Cloud.

### **Problem: "Module not found"**

**Solution:** Check `requirements.txt` includes all packages:

streamlit==1.31.0  
pandas==2.1.4  
numpy==1.26.3  
scikit-learn==1.4.0  
plotly==5.18.0

### **Problem: App crashes on startup**

**Solution:**

1. Check Streamlit Cloud logs (bottom of deploy page)  
2. Look for specific error message  
3. Common issues:  
   * Missing .pkl file ← Most common  
   * Wrong Python version  
   * Dependency conflicts

### **Problem: Large .pkl file**

**Solution:** If GitHub rejects the file (\>100MB):

\# Use Git LFS for large files  
git lfs install  
git lfs track "\*.pkl"  
git add .gitattributes  
git add clustering\_models.pkl  
git commit \-m "Add models with LFS"  
git push

---

## **🎉 Success\!**

Once deployed, you can:

✅ Share your live app URL with anyone ✅ Add it to your resume/portfolio ✅ Show it in interviews ✅ Include it in your GitHub profile README

Example URLs to add to your portfolio:

* **Live App:** https://your-app.streamlit.app  
* **GitHub Repo:** https://github.com/yourusername/customer-segmentation-clustering  
* **Kaggle Dataset:** https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

---

## **📱 Sharing Your Project**

### **LinkedIn Post Template**

🎯 Excited to share my latest Machine Learning project\!

Built an interactive dashboard comparing K-Means and DBSCAN clustering algorithms on customer segmentation data.

🔹 Features:  
\- Interactive visualizations with Plotly  
\- Real-time parameter tuning  
\- Comprehensive algorithm comparison  
\- Business insights and recommendations

🛠️ Tech Stack: Python, Scikit-learn, Streamlit, Plotly

🔗 Live Demo: \[Your Streamlit URL\]  
💻 Source Code: \[Your GitHub URL\]

\#MachineLearning \#DataScience \#Clustering \#Python \#Streamlit

---

## **🔄 Making Updates**

After deployment, to update your app:

\# Make changes to app.py or other files  
\# Then:  
git add .  
git commit \-m "Description of changes"  
git push

\# Streamlit Cloud auto-detects changes and redeploys\!

---

## **🆘 Need Help?**

* **Streamlit Docs:** https://docs.streamlit.io  
* **GitHub Issues:** Create an issue in your repository  
* **Streamlit Community:** https://discuss.streamlit.io

---

**🎊 Congratulations\! You've successfully deployed your clustering dashboard\!** 🎊

