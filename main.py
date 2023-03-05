import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

# -------------
# layout design
# -------------

st.title('Country GDP Estimation Tool')
st.write('''
         Please fill in the attributes below, then hit the GDP Estimate button
         to get the estimate. 
         ''')

st.header('Input Attributes')
att_regn = st.selectbox('Region', options=(1, 2, 3, 4, 5, 6, 7))
st.write('''
         * 1 : Latin America & Caribbean
         * 2 : South Asia
         * 3 : Sub-Saharan Africa
         * 4 : Europe & Central Asia
         * 5 : Middle East & North Africa
         * 6 : East Asia & Pacific
         * 7 : North America
         '''
         )

if att_regn == 1:
    att_regn_1 = 1
    att_regn_2 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = 0
elif att_regn == 2:
    att_regn_2 = 1
    att_regn_1 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = 0
elif att_regn == 3:
    att_regn_3 = 1
    att_regn_1 = att_regn_2 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = 0
elif att_regn == 4:
    att_regn_4 = 1
    att_regn_1 = att_regn_3 = att_regn_2 = att_regn_5 = att_regn_6 = att_regn_7 = 0
elif att_regn == 5:
    att_regn_5 = 1
    att_regn_1 = att_regn_3 = att_regn_4 = att_regn_2 = att_regn_6 = att_regn_7 = 0
elif att_regn == 6:
    att_regn_6 = 1
    att_regn_1 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_2 = att_regn_7 = 0
else:
    att_regn_7 = 1
    att_regn_1 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_2 = 0

att_area = st.slider('Area (Sq. Km.)', min_value=1000.0, max_value=17e6, value=6e5, step=1e4)
att_popl = st.number_input('Population (Example: 10000000)', min_value=1e4, max_value=2e9, value=2e7)
att_grow = st.slider('Population growth (%)', min_value=-10.0, max_value=10.0, value=0.0, step=0.01)
att_dens = st.number_input('Population Density (Sq. Km.)', min_value=10.0, max_value=30000.00, value=100.0)
att_lite = st.slider('Literacy Rate (%)', min_value=0.0, max_value=100.0, value=50.0)
att_life = st.slider('Life Expectancy (Years)', min_value=0.0, max_value=100.0, value=50.0)
att_brth = st.slider('Birth Rate (Per 1000 People)', min_value=0.0, max_value=100.0, value=50.0)
att_deth = st.slider('Death Rate (Per 1000 People)', min_value=0.0, max_value=100.0, value=50.0)
att_mort = st.slider('Infant Mortality (Per 1000 Births)', min_value=0.0, max_value=100.0, value=50.0)
att_urba = st.slider('Urban Population (% of Total Population)', min_value=0.0, max_value=100.0, value=50.0)
att_rura = st.slider('Rural Population (% of Total Population)', min_value=0.0, max_value=100.0, value=50.0)
att_unem = st.slider('Unemployment, Total (% of Total Labor)', min_value=0.0, max_value=100.0, value=50.0)
att_inte = st.slider('Internet Users (% of Total Population)', min_value=0.0, max_value=100.0, value=50.0)
att_phon = st.slider('Phones Per 1000 Person', min_value=0.0, max_value=1000.0, value=500.0)
att_elec = st.slider('Access to Electricity (% of Population)', min_value=0.0, max_value=100.0, value=50.0)
att_fore = st.slider('Forest Area (% of Total Area)', min_value=0.0, max_value=100.0, value=50.0)
att_agri = st.slider('Agricultural Land Area (% of Total Area)', min_value=0.0, max_value=100.0, value=50.0)
att_infl = st.slider('Consumer Inflation (Annual %)', min_value=0.0, max_value=1000.0, value=500.0)
att_wome = st.slider('Women Seats In Parliament (%)', min_value=0.0, max_value=100.0, value=50.0)
att_poli = st.slider('Political Stability And Absence Of Violence/Terrorism(%)', min_value=0.0, max_value=100.0, value=50.0)
att_huma = st.slider('Human Capital Index', min_value=0.0, max_value=1.0, value=0.5)
att_migr = st.slider('Annual Net Migration (migrant(s)/1,000 population)', min_value=-100000, max_value=100000, value=0, step=1)
att_debt = st.slider('Government Debt (% of GDP)', min_value=0.0, max_value=1000.0, value=500.0)
att_mili = st.slider('Military Expenditure (% of GDP)', min_value=0.0, max_value=100.0, value=50.0)
att_educ = st.slider('Education Expenditure (% of GDP)', min_value=0.0, max_value=100.0, value=50.0)
att_remi = st.slider('Remittances (% of GDP)', min_value=0.0, max_value=100.0, value=50.0)
att_heal = st.slider('Health Expenditure (% of GDP)', min_value=0.0, max_value=100.0, value=50.0)
att_agco = st.slider('Agriculture Contribution (% of GDP)', min_value=0.0, max_value=100.0, value=50.0)
att_indu = st.slider('Industry Contribution (% of GDP)', min_value=0.0, max_value=100.0, value=50.0)
att_serv = st.slider('Service Contribution (% of GDP)', min_value=0.0, max_value=100.0, value=50.0)


user_input = np.array([att_popl, att_area, att_dens, att_grow, att_lite, att_life, att_brth, att_deth,
                       att_mort, att_urba, att_rura , att_unem, att_inte, att_phon, att_elec, att_fore,
                       att_agri, att_infl, att_wome, att_poli, att_huma, att_migr, att_debt, att_mili,
                       att_educ, att_remi, att_heal, att_agco, att_indu, att_serv, att_regn_1, att_regn_2, att_regn_3,
                       att_regn_4, att_regn_5, att_regn_6, att_regn_7]).reshape(1, -1)


# ------
# Model
# ------

# import dataset
def get_dataset():
    data = pd.read_csv('GDPdata.csv')
    return data


if st.button('Estimate GDP'):
    data = get_dataset()

    # fix column names
    data.columns = (["Country", "Region", "Area (sq. km.)", "Population", "Population growth (%)",
                "Pop. density (per sq. km.)", "Literacy rate (%)", "Life expectancy (years)",
                "Birth rate (per 1000)", "Death rate  (per 1000)", "Infant mortality (per 1000 births)",
                "Urban population (% of total population)", "Rural population (% of total population)",
                "Unemployment, total (% of total labor)", "Phones (per 100 people)", "Internet Users (% population)",
                "Access to electricity (% of population)", "Forest area (% of land area)", "Agricultural land (% of land area)",
                "Inflation (annual %)", "Women Seats In Parliament (%)", "Political Stability And Absence Of Violence/Terrorism(%)",
                "Human Capital Index", "Net migration", "Govt. Debt (% of GDP)", "Military expenditure (% of GDP)",
                "Education expenditure (% of GDP)", "Health expenditure (% of GDP)",
                "Agriculture contribution (% of GDP)", "Industry contribution (% of GDP)", "Service contribution (% of GDP)",
                "Remittances (% of GDP)", "GDP (current US$)", "GDP per capita (current US$)", "GDP growth (annual %)"])


    # Region Transform
    data_final = pd.concat([data, pd.get_dummies(data['Region'], prefix='Region')], axis=1).drop(['Region'], axis=1)

    # Data Split
    y = data_final['GDP per capita (current US$)']
    X = data_final.drop(['GDP per capita (current US$)', 'Country'], axis=1)
    # split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # model training
    gbm_opt = GradientBoostingRegressor(learning_rate=0.01, n_estimators=500,
                                        max_depth=5, min_samples_split=10,
                                        min_samples_leaf=1, subsample=0.7,
                                        max_features=7, random_state=101)
    gbm_opt.fit(X_train, y_train)

    # making a prediction
    gbm_predictions = gbm_opt.predict(user_input)  # user_input is taken from input attrebutes
    st.write('The estimated GDP per capita is: ', gbm_predictions)


