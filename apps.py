import streamlit as st
import pandas as pd
import pickle

# =========================================================
# LOAD MODEL
# =========================================================
with open("course_completion_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🎓 Prediksi Penyelesaian Kursus Online")
st.write("Masukkan data peserta untuk melihat apakah mereka akan menyelesaikan course atau tidak.")
st.divider()

# =========================================================
# INPUT TAMBAHAN
# =========================================================
NamaPeserta = st.text_input("Nama Student Course")

# =========================================================
# INPUT FEATURES
# =========================================================

TimeSpentOnCourse = st.number_input("Total waktu yang dihabiskan (jam)", min_value=0.0)
NumberOfVideosWatched = st.number_input("Jumlah video yang ditonton", min_value=0)
NumberOfQuizzesTaken = st.number_input("Jumlah kuis yang dikerjakan", min_value=0)
QuizScores = st.number_input("Rerata skor kuis (0–100)", min_value=0.0, max_value=100.0)
CompletionRate = st.number_input("Completion rate (%)", min_value=0.0, max_value=100.0)

# DEVICE TYPE
DeviceType = st.selectbox("Device yang digunakan", ["Desktop", "Mobile"])
DeviceType = 0 if DeviceType == "Desktop" else 1

# COURSE CATEGORY (OHE inside pipeline)
CourseCategory = st.selectbox(
    "Kategori Kursus",
    ["Arts", "Business", "Health", "Programming", "Science"]
)

# =========================================================
# SUSUN INPUT SESUAI KOLOM DATASET RAW
# =========================================================

input_data = pd.DataFrame([{
    "TimeSpentOnCourse": TimeSpentOnCourse,
    "NumberOfVideosWatched": NumberOfVideosWatched,
    "NumberOfQuizzesTaken": NumberOfQuizzesTaken,
    "QuizScores": QuizScores,
    "CompletionRate": CompletionRate,
    "CourseCategory": CourseCategory,
    "DeviceType": DeviceType
}])

# =========================================================
# PREDIKSI
# =========================================================

if st.button("🔍 Prediksi Kelulusan"):
    pred = model.predict(input_data)[0]

    st.subheader("Hasil Prediksi")
    if pred == 1:
        st.success("🎉 Peserta diprediksi **MENYELESAIKAN** kursus!")
    else:
        st.error("❌ Peserta diprediksi **TIDAK MENYELESAIKAN** kursus.")