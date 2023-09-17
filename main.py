import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
import streamlit as st
from sklearn.metrics import r2_score

st.set_page_config(
        page_title="My Page Title",
        layout="wide",
)

st.header('📈 Автоматическая модель линейной регрессии', divider='red')
uploaded_file = st.file_uploader('Choose a file')
if uploaded_file is not None:
        ft = uploaded_file.type.split('/')[1]
        if ft == 'csv':
                df = pd.read_csv(uploaded_file)
        elif ft == 'xls' or ft == 'xlsx':
                df = pd.read_excel(uploaded_file)
        df.dropna(inplace=True)
        st.text('Первые три строчки вашего датасета:')
        st.dataframe(df.head(3))
        #df.apply(LabelEncoder().fit_transform)
        target = st.selectbox(
                'Выберите столбец Y',
                list(df.columns)
        )
        target_num = 0
        for i in range(len(df.columns)):
                if target == df.columns[i]:
                        target_num = i
        y = df.iloc[:,target_num]
        st.dataframe(y)
        #y = df.iloc[:, -1]
        c = df.columns[-1]
        x = df.drop(target, axis=1)
        options = st.multiselect(
                'Уберите столбцы, которые не смогут помочь предсказать результат. Пример: ID объекта',
                x.columns)
        x = x.drop(options, axis=1)
        options1 = st.multiselect(
                'Выберите столбцы с категориальными признаками, чтобы перевести их в численный вид',
                x.columns)
        le = LabelEncoder()
        for i in options1:
                x[i] = le.fit_transform(x[i])
        st.dataframe(x)
        #options2 = st.multiselect(
                #'Выберите столбцы к которым хотите применить регуляризацию',
                #x.columns)
        #ss = StandardScaler()
        #x = ss.fit_transform(x)
        #for i in options2:
                #x[i] = ss.fit_transform(x[i])
        cols = list(x.columns)
        st.text('X данные')
        st.dataframe(x)
        st.text('Y данные')
        st.dataframe(y)
        X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        modeldt = DecisionTreeRegressor(random_state=42)
        modeldt.fit(X_train, y_train)
        y_pred_dt = modeldt.predict(X_test)

        ridgeReg = Ridge(random_state=42)
        ridgeReg.fit(X_train, y_train)
        y_pred_ridge = ridgeReg.predict(X_test)


        fig = plt.figure(figsize=(12, 12))

        nm = r2_score(y_test, y_pred)
        quality = round(nm * 100, 2)
        q_i = str(quality)
        st.text(f'Точность прогноза (линейная регрессия):{q_i}%')

        nm1 = r2_score(y_test, y_pred_dt)
        quality1 = round(nm1 * 100, 2)
        q_i1 = str(quality1)
        st.text(f'Точность прогноза (дерево решений):{q_i1}%')

        nm2 = r2_score(y_test, y_pred_ridge)
        quality2 = round(nm2 * 100, 2)
        q_i2 = str(quality2)
        st.text(f'Точность прогноза (Ridge):{q_i2}%')

        plt.subplot(3,1,1)
        plt.scatter(X_test.index, y_test, color="red")
        plt.scatter(X_test.index, y_pred, color="blue")
        plt.subplot(3, 1, 2)
        plt.scatter(X_test.index, y_test, color="red")
        plt.scatter(X_test.index, y_pred_dt, color="blue")
        st.pyplot(fig)
        plt.subplot(3, 1, 3)
        plt.scatter(X_test.index, y_test, color="red")
        plt.scatter(X_test.index, y_pred_ridge, color="blue")
        st.pyplot(fig)

        lst = []
        for i in range(len(cols)):
                line = str(cols[i])
                a = st.number_input(line)
                lst.append(a)
        for i in range(len(lst)):
                st.text(lst[i])
        st.text(lst)
        test = pd.DataFrame(lst, cols)
        test = test.T
        pred = model.predict(test)
        st.text(f'Predicted {c} = {pred}')
else:
        st.warning('You need to upload a csv or excel file.')
