# 🎓 Online Course Recommendation System
Check Out My Project Dashboard - https://project-online-course-recommendation-system-95vqkdguoi3czscjyf.streamlit.app/
## 🧠 Project Overview
The **Online Course Recommendation System** is a **Data Science and Machine Learning project** that helps learners find the most suitable online courses based on their interests, skills, and preferences.  
It uses **Natural Language Processing (NLP)** and **Machine Learning** to recommend similar courses by analyzing the dataset features such as course name, category, difficulty, instructor, and ratings.

## 🚀 Objectives
- Recommend relevant online courses to users based on their input or course name.  
- Use **TF-IDF Vectorization** and **Cosine Similarity** to analyze and compare courses.  
- Build an interactive **Streamlit web app** for real-time course recommendations.

## 📊 Dataset Description
The dataset contains information about online courses including:

| Column Name | Description |
|--------------|-------------|
| `course_id` | Unique ID of the course |
| `course_name` | Title of the course |
| `instructor` | Name of the instructor |
| `course_duration_hours` | Duration of the course in hours |
| `difficulty` | Level of the course (Beginner / Intermediate / Advanced) |
| `rating` | Course rating (out of 5) |
| `course_price` | Price of the course |
| `category` | Category or domain of the course |

## ⚙️ Tech Stack
- **Language:** Python  
- **Libraries Used:**  
  - `pandas`, `numpy` – Data handling and preprocessing  
  - `scikit-learn` – Machine Learning, TF-IDF, Cosine Similarity  
  - `streamlit` – Web app development  
  - `joblib` – Model persistence  
  - `matplotlib` / `seaborn` – Visualization (optional)

## 🧮 Methodology
1. **Data Preprocessing:**  
   Cleaned dataset, handled null values, and selected important features.

2. **Feature Engineering:**  
   Combined text-based features (e.g., `course_name` + `category`) and applied **TF-IDF Vectorization**.

3. **Model Building:**  
   Used **Cosine Similarity** to calculate similarity scores and recommend the top N similar courses.

4. **Deployment:**  
   Built an interactive **Streamlit Dashboard** to allow users to input a course name and get recommended results.

## 🖥️ Streamlit App Features
- 📂 Upload dataset (CSV file)  
- 🔍 Search for a course name or keyword  
- 📊 Display top similar courses with key details  
- ⭐ Show instructor, difficulty, and rating information  

