import numpy as np
import pandas as pd
import streamlit as st
import pickle
import xgboost as xgb

st.markdown(
    """
    <style>
    [data-testid="stAppViewBlockContainer"]{
        max-width: 950px !important;
        margin: 25px auto !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 600px !important;
        }
         </style>
    """,
    unsafe_allow_html=True,
)

df_model = pd.read_csv('model_set.csv', index_col=0)
df_localize = pd.read_csv("local_set.csv", index_col=0)
df_localize = df_localize.drop_duplicates(subset="Refnis code")
df_localize["Locality"] = df_localize["Locality"].apply(lambda x: x.title())

conversion_dict_house = {"Walloon Brabant": 2465, "Hainaut": 1376, "Namur": 1601, "Liège": 1682, "Luxembourg": 1616, "Brussels": 3241, 
                                 "Flemish Brabant": 2465, "West Flanders": 2055, "East Flanders": 2189, "Antwerp": 2343, "Limburg": 1865}
conversion_dict_apt = {"Walloon Brabant": 3139, "Hainaut": 1840, "Namur": 2423, "Liège": 2214, "Luxembourg": 2427, "Brussels": 3370, 
                               "Flemish Brabant": 3159, "West Flanders": 3803, "East Flanders": 2845, "Antwerp": 2732, "Limburg": 2482}
convert_kitchen = {'Unknown': np.nan, 'Uninstalled': 1, 'USA uninstalled': 2, 'Installed': 3, 'USA installed': 4, 'Semi-equipped': 5, 
                   'USA semi-equipped': 6, 'Hyper-equipped': 7, 'USA hyper-equipped': 8}
convert_building = {'Unknown': np.nan, 'To be done up': 1, 'To restore': 2, 'To renovate': 3, 'Good': 4, 'Just renovated': 5, 'As new': 6}
convert_PEB = {'Unknown': np.nan, 'G': 1, 'F': 2, 'E': 3, 'D': 4, 'C': 5, 'B': 6, 'A': 7}
    
def get_density(locality):
    return int(df_localize.loc[df_localize["Locality"] == locality, "Density"].values[0])

def get_revenue(locality):
    return int(df_localize.loc[df_localize["Locality"] == locality, "Median_revenue"].values[0])

def replace_nan(dict, key, new_value):
    if dict[key] == np.nan:
        dict[key] = new_value
    return dict    

def get_user_input():
    with st.sidebar:
        st.header("Your property information")
        
        Property = {}
        
        TypeProperty = st.selectbox("Type of property", ("House", "Apartment"))
        Property["TypeOfProperty_numerical"] = 1 if TypeProperty == "Apartment" else 2
        
        Region = st.selectbox("Region", ("Wallonie", "Brussels", "Flanders"))
        
        Province = st.selectbox("Province", (df_localize[df_localize["Region"] == Region]["Province"].unique()))
        if Property["TypeOfProperty_numerical"] == 1:
            Property["Province_numerical"] = int(conversion_dict_apt[Province])
        else:
            Property["Province_numerical"] = int(conversion_dict_house[Province])
        
        Locality = st.selectbox("Locality", (df_localize[df_localize["Province"] == Province]["Locality"].unique()))
        Density = get_density(Locality)
        Median_revenue = get_revenue(Locality)
        Property["Density"] = int(Density)
        Property["Median_revenue"] = int(Median_revenue)        

        LivingArea = st.text_input("Living area (in m²)", value=0, placeholder="Enter living area in m²")
        error_message_1 = ""
        try:
            LivingArea = int(LivingArea)
            Property["LivingArea"] = LivingArea
        except ValueError:
            error_message_1 = "<p style='color: red; font-size: 16px;'>Please enter a valid number for 'Living area'</p>"
            st.markdown(error_message_1, unsafe_allow_html=True)

        SurfacePlot = st.text_input("Surface of the plot (in m²)", value=0, placeholder="Enter surface of the plot in m²")
        error_message_2 = ""
        try:
            SurfacePlot = int(SurfacePlot)
            Property["SurfaceOfPlot"] = SurfacePlot
        except ValueError:
            error_message_2 = "<p style='color: red; font-size: 16px;'>Please enter a valid number for 'Surface of the plot'</p>"
            st.markdown(error_message_2, unsafe_allow_html=True)
            
        NumberOfFacades = st.radio("Number of facades", (2, 3, 4, "Not applicable"), key="facades")
        Property["NumberOfFacades"] = int(NumberOfFacades) if NumberOfFacades != "Not applicable" else 0
        
        BedroomCount = st.number_input("Number of bedrooms", 0, 15)
        Property["BedroomCount"] = int(BedroomCount)
        
        BathroomCount = st.number_input("Number of bathrooms", 0, 10)
        Property["BathroomCount"] = int(BathroomCount)
        
        KitchenType = st.selectbox("Kitchen", ("USA hyper-equipped", "Hyper-equipped", "USA semi-equipped", "Semi-equipped", "USA installed", "Installed", "Uninstalled", "USA uninstalled", "Unknown"))
        Property["Kitchen_numerical"] = convert_kitchen[KitchenType]
        
        Garden = st.radio("Garden", ("Yes", "No"), key="garden")
        Property["Garden"] = 0 if Garden == "No" else 1
        
        Terrace = st.radio("Terrace", ("Yes", "No"), key="terrace")
        Property["Terrace"] = 0 if Terrace == "No" else 1
        
        SwimmingPool = st.radio("Swimming pool", ("Yes", "No"), key="pool")
        Property["SwimmingPool"] = 0 if SwimmingPool == "No" else 1
        
        ConstructionYear = st.slider("Year of construction", 1800, 2028, 2024)
        Property["ConstructionYear"] = int(ConstructionYear)
        
        StateBuilding = st.selectbox("State of the building", ("As new", "As renovated", "Good", "To renovate", "To restore", "To be done up", "Unknown"))
        Property["StateOfBuilding_numerical"] = convert_building[StateBuilding]
        
        PEB = st.selectbox("Score PEB", ("A", "B", "C", "D", "E", "F", "G", "Unknown"))
        Property["PEB_numerical"] = convert_PEB[PEB]     
        
        for key in ["Kitchen_numerical", "StateOfBuilding_numerical", "PEB_numerical"]:
            replace_nan(Property, key, df_model[key].mode()[0])   
                    
        df_user = pd.DataFrame([Property])
        df_user = df_user.reindex(columns=df_model.columns)
        df_user_array = np.array(df_user)
        
        st.write(df_user.columns)

        st.write(df_user.columns)
        
        Property_display = {"Type Of Property": TypeProperty, "Region": Region, "Province": Province, "Locality": Locality, "Surface of the plot": SurfacePlot, "Number of facades": NumberOfFacades, 
                            "Living area": LivingArea, "Number of bedrooms": BedroomCount, "Number of bathrooms": BathroomCount, "Garden": Garden, "Terrace": Terrace, "Swimming Pool": SwimmingPool, 
                            "Kitchen": KitchenType, "Year of construction": ConstructionYear, "State of the building": StateBuilding, "PEB": PEB}
        
        st.markdown("""
            <style>
            .stButton button {
                font-size: 20px;
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                position: relative;
            }
            </style>
            """, unsafe_allow_html=True)
        
        button_submit = st.button("Submit your information")
        
        if error_message_1 or error_message_2:
            return None, None, None 
            
        return df_user_array, Property_display, button_submit
        
def main():
    st.title('ImmoEliza - Get the most accurate estimation for your property !')
    
    df_user_array, Property_display, button_submit = get_user_input()
    
    if button_submit:
        st.subheader('Summary of your property features')
        df_display = pd.DataFrame(list(Property_display.items()), columns=['Category', 'Your property'])
        df_display.set_index('Category', inplace=True)
        st.table(df_display)
    
        xgb_model = pickle.load(open('xgb_model.pkl', 'rb'))
        prediction = xgb_model.predict(df_user_array)
    
        st.subheader('Your estimation')
        formatted_prediction = f"{int(prediction[0]):,}".replace(',', '.')
        st.markdown(f'<p style="font-size: 22px;">The estimated price of your property is <strong>{formatted_prediction} €</strong></p>', unsafe_allow_html=True)
    
    else:
        st.write("")
        st.markdown("<p style='font-size: 22px;'>ImmoEliza is a new machine learning project aiming at revolutionizing real estate estimations.</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 22px;'>Based on the most recent and accurate data about real estate properties in Belgium, ImmoEliza delivers the most precise estimations.</p>", unsafe_allow_html=True)
        st.markdown("### <span style='color: grey;'>Best Real Estate Estimator</span> for the third year in a row at the 25th World House Pricing Olympics in Chaumes-sur-Marne", unsafe_allow_html=True)
        st.image("House_money.jpg", use_column_width=True)  
    
if __name__ == "__main__":
    main()