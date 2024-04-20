import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

def get_Data():
    input= [1.799e+01, 1.038e+01, 1.228e+02, 1.001e+03, 1.184e-01, 2.776e-01,
       3.001e-01, 1.471e-01, 2.419e-01, 7.871e-02, 1.095e+00, 9.053e-01,
       8.589e+00, 1.534e+02, 6.399e-03, 4.904e-02, 5.373e-02, 1.587e-02,
       3.003e-02, 6.193e-03, 2.538e+01, 1.733e+01, 1.846e+02, 2.019e+03,
       1.622e-01, 6.656e-01, 7.119e-01, 2.654e-01, 4.601e-01, 1.189e-01]
    return input

def get_sliderData():
    summary_df = pickle.load(open('models/summary.pkl','rb'))

    feature_names = [
        "Radius Mean",
        "Texture Mean",
        "Perimeter Mean",
        "Area Mean",
        "Smoothness Mean",
        "Compactness Mean",
        "Concavity Mean",
        "Concave Points Mean",
        "Symmetry Mean",
        "Fractal Dimension Mean",
        "Radius SE",
        "Texture SE",
        "Perimeter SE",
        "Area SE",
        "Smoothness SE",
        "Compactness SE",
        "Concavity SE",
        "Concave Points SE",
        "Symmetry SE",
        "Fractal Dimension SE",
        "Radius Worst",
        "Texture Worst",
        "Perimeter Worst",
        "Area Worst",
        "Smoothness Worst",
        "Compactness Worst",
        "Concavity Worst",
        "Concave Points Worst",
        "Symmetry Worst",
        "Fractal Dimension Worst"
    ]

    summary_df['Name'] = feature_names

    return summary_df

def get_scaledData(input):
    with open('models/scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    
    input_array = np.array(input)
    input_array = input_array.reshape(1, -1)
    input_df = pd.DataFrame(input_array)
    return scaler.transform(input_df)

def get_prediction(scaled):
    with open('models/lrmodel.pkl', 'rb') as file:
        model = pickle.load(file)
    res = model.predict(scaled)

    st.write("Probability of being Benign", model.predict_proba(scaled)[0][0])
    st.write("Probability of being Malignant", model.predict_proba(scaled)[0][1])
    return res[0]

def get_radarChart(outputs):
    categories = ["Radius","Texture","Perimeter","Area","Smoothness","Compactness","Concavity","Concave Points","Symmetry","Fractal Dimension"]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[outputs['radius_mean'], outputs['texture_mean'], outputs['perimeter_mean'], outputs['area_mean'],outputs['smoothness_mean'],outputs['compactness_mean'],outputs['concavity_mean'],outputs['concave points_mean'],outputs['symmetry_mean'],outputs['fractal_dimension_mean']],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[outputs['radius_se'], outputs['texture_se'], outputs['perimeter_se'], outputs['area_se'],outputs['smoothness_se'],outputs['compactness_se'],outputs['concavity_se'],outputs['concave points_se'],outputs['symmetry_se'],outputs['fractal_dimension_se']],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[outputs['radius_worst'], outputs['texture_worst'], outputs['perimeter_worst'], outputs['area_worst'],outputs['smoothness_worst'],outputs['compactness_worst'],outputs['concavity_worst'],outputs['concave points_worst'],outputs['symmetry_worst'],outputs['fractal_dimension_worst']],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]
        )),
    showlegend=True
    )

    return fig

def add_sidebar(inputs):
    st.sidebar.header("Cell Neuclei Measurement")

    summary_df = get_sliderData()

    output_dict = {}

    for (index,row),i  in zip(summary_df.iterrows(),inputs):
        output_dict[row['Variable Name']] = st.sidebar.slider( label=row['Name'], min_value=row['Min'], max_value=row['Max'],value=i)
    
    return output_dict

def main():
    st.set_page_config(
        page_title="Breast Cancer Prediction",
        page_icon="female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    inputs = get_Data()

    outputs = add_sidebar(inputs)

    with st.container():
        st.title("Breast Cancer Prediction")
        st.write("Please connect this app")

    col1, col2 = st.columns([4,1])

    with col1:
        fig = get_radarChart(outputs)
        st.plotly_chart(fig)

    with col2:
        prediction_result = st.write(get_prediction(get_scaledData(get_Data())))
        if prediction_result ==0:
            res = "Benign"
        else:
            res = "Malignant"
            # Determine background color based on prediction result
        if prediction_result == 0:
            # Green background if prediction_result is 0
            bg_color = 'green'
        else:
            # Red background if prediction_result is not 0
            bg_color = 'red'

        # Define the HTML content for styled div with conditional background color
        styled_div_html = f"""
            <div style="background-color: {bg_color}; border-radius: 15px; padding: 20px;">
                Result: {res}
            </div>
        """

        # Display the styled div element with conditional styling
        st.markdown(styled_div_html, unsafe_allow_html=True)

if __name__== "__main__" :
    main()
