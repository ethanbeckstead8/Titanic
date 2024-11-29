import streamlit as st
from sklearn.datasets import fetch_openml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

@st.cache_data
def loadData():
    titanic_sklearn = fetch_openml('titanic', version=1, as_frame=True)
    return titanic_sklearn.frame

titanic = loadData()
st.title("Ethan Beckstead's Stream LIT app for Titanic")
st.markdown("Go to the side to filter the data. Below on tab 1 we have a graphic and on tab 2 we have a filtered table and summary statistic")

st.sidebar.header("Filter Data Here")
sex = st.sidebar.selectbox("Sex", ["All", "Male", "Female"])
age = st.sidebar.slider("Age Range", int(titanic['age'].min()), int(titanic['age'].max()), (0, 80))
pclass = st.sidebar.multiselect("Passenger Classes", titanic['pclass'].unique(), default=titanic['pclass'].unique())
embarked = st.sidebar.radio("Port of Embarkation", ["All", "C", "Q", "S"])

data = titanic[
    ((titanic['sex'].str.capitalize() == sex) | (sex == "All")) &
    (titanic['age'].between(age[0], age[1], inclusive="both")) &
    (titanic['pclass'].isin(pclass)) & 
    ((titanic['embarked'] == embarked) | (embarked == "All"))
]

tab1, tab2 = st.tabs(["Graphic", "Summary Statistic and Filtered Table"])

with tab1:
    st.header("Graphic")
    st.subheader("Histogram of Age Distribution based on Filtered Data")
    fig, ax = plt.subplots()
    sns.histplot(data['age'].dropna(), kde=True, ax=ax, color="royalblue")
    ax.set_title("Age Distribution of Titanic Passengers")
    ax.set_xlabel("Age")
    ax.set_ylabel("Number of Passengers")
    st.pyplot(fig)

with tab2:
    st.header("Summary Statistic and Filtered Table")
    data['survived'] = data['survived'].astype(int)
    zeros = data['survived'].value_counts().get(0, 0)
    ones = data['survived'].value_counts().get(1, 0)
    survival_rate = 100 * ones / (zeros + ones)
    st.write(f"Survival Rate: {survival_rate:.2f}%")


    st.write("Filtered Table (For verification of Survival Rate calculations and Filters)", data)

st.markdown("Data from SKLearn (Titanic Dataset)")
