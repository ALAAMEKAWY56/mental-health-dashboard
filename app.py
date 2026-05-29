import streamlit as st
import pandas as pd
import plotly.io as pio
import plotly.express as px
import joblib
import numpy as np
import base64
import time
import os

st.set_page_config(
    page_title="Mental Health EDA Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

pio.templates["mental_health_theme"] = dict(
    layout=dict(
        paper_bgcolor="#F7F3ED",
        plot_bgcolor="#FFFDF9",
        font=dict(family="Arial", color="#3E2C23", size=14),
        title=dict(font=dict(size=22, color="#6F4E37"), x=0.5),
        colorway=[
            "#C8A97E", "#D9B382", "#E89B5B", "#A47551",
            "#8B5E3C", "#DDB892", "#B08968", "#9C6644",
            "#B56576", "#9D4E4E", "#7F3B3B", "#E6CCB2"
        ],
        xaxis=dict(
            gridcolor="#E5D7C8",
            zerolinecolor="#E5D7C8",
            linecolor="#D6CCC2",
            tickfont=dict(color="#3E2C23")
        ),
        yaxis=dict(
            gridcolor="#E5D7C8",
            zerolinecolor="#E5D7C8",
            linecolor="#D6CCC2",
            tickfont=dict(color="#3E2C23")
        ),
        legend=dict(
            bgcolor="#FFFDF9",
            bordercolor="#E5D7C8",
            borderwidth=1,
            font=dict(color="#3E2C23")
        ),
        hoverlabel=dict(
            bgcolor="#FFFDF9",
            font_size=13,
            font_family="Arial"
        )
    )
)

pio.templates.default = "mental_health_theme+plotly_white"

# =========================================
# GLOBAL CSS STYLING
# =========================================
st.markdown(
    """
    <style>

    /* =========================================
    FORCE LIGHT MODE
    ========================================= */
    :root {
        color-scheme: light !important;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background-color: #F7F3ED !important;
        color: #3E2C23 !important;
    }

    [data-testid="stAppViewContainer"] > div {
        background-color: #F7F3ED !important;
    }
    .stApp {
        background-color: #F7F3ED;
        color: #3E2C23;
        font-family: Arial, sans-serif;
    }

    .modebar {
        background-color: transparent !important;
    }

    .modebar-btn svg {
        fill: #6F4E37 !important;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    header[data-testid="stHeader"] {
        background-color: #F7F3ED !important;
        border-bottom: 1px solid #E5D7C8;
    }

    [data-testid="stToolbar"] {
        right: 2rem;
    }

    [data-testid="stDecoration"] {
        background: #F7F3ED;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    button[kind="header"] {
        background-color: #FFFDF9 !important;
        color: #6F4E37 !important;
        border-radius: 12px !important;
        border: 1px solid #E5D7C8 !important;
        transition: 0.3s ease;
    }

    button[kind="header"]:hover {
        background-color: #E6CCB2 !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #FFFDF9;
        border-right: 1px solid #E5D7C8;
    }

    section[data-testid="stSidebar"] * {
        color: #3E2C23;
    }

    h1, h2, h3 {
        color: #6F4E37;
        font-weight: 700;
    }

    p, label, span {
        color: #3E2C23;
    }

    div[data-testid="stMetric"] {
        background-color: #FFFDF9;
        padding: 18px;
        border-radius: 18px;
        border: 1px solid #E5D7C8;
        box-shadow: 0px 4px 12px rgba(111, 78, 55, 0.08);
    }

    .stButton > button {
        background-color: #A47551;
        color: #FFFDF9;
        border-radius: 12px;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #7F3B3B;
        color: #FFFDF9;
    }

    div[data-baseweb="select"] > div {
        background-color: #FFFDF9;
        border-radius: 12px;
        border: 1px solid #E5D7C8;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 14px;
        border: 1px solid #E5D7C8;
        background-color: #FFFDF9;
    }

    div[data-testid="stPlotlyChart"] {
        background-color: #F7F3ED !important;
        border-radius: 22px;
        padding: 16px;
        border: 1px solid #E5D7C8;
        box-shadow:
            0 0 8px rgba(232, 155, 91, 0.18),
            0 0 18px rgba(181, 101, 118, 0.12),
            0 0 28px rgba(127, 59, 59, 0.08);
        overflow: hidden !important;
    }

    div[data-testid="stPlotlyChart"] > div {
        width: 100% !important;
    }

    hr {
        border: none;
        height: 1px;
        background-color: #E5D7C8;
    }

    div[role="radiogroup"] {
        gap: 4px;
    }

    div[role="radiogroup"] label {
        background-color: #f5e8e1 !important;
        border: 1px solid #D6B89E !important;
        border-radius: 14px !important;
        padding: 12px 16px !important;
        margin-bottom: 6px !important;
        min-height: 52px !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        box-shadow: 0px 3px 8px rgba(111, 78, 55, 0.05);
        transition: all 0.25s ease;
    }

    div[role="radiogroup"] label:hover {
        background-color: #E7D3C1 !important;
        border-color: #C8A97E !important;
        transform: translateY(-1px);
        box-shadow: 0px 5px 12px rgba(111, 78, 55, 0.10);
    }

    div[role="radiogroup"] label[data-selected="true"] {
        background-color: #A47551 !important;
        border-color: #A47551 !important;
        box-shadow: 0px 6px 16px rgba(111, 78, 55, 0.16);
    }

    div[role="radiogroup"] label[data-selected="true"] p {
        color: white !important;
        font-weight: 700 !important;
    }

    div[role="radiogroup"] p {
        color: #4E342E;
        font-size: 14px;
        font-weight: 600;
        margin: 0;
    }

    .kpi-card {
        background-color: #FFFDF9;
        border: 1px solid #E5D7C8;
        border-radius: 22px;
        padding: 22px 18px;
        text-align: center;
        box-shadow: 0px 4px 12px rgba(111, 78, 55, 0.08);
        transition: 0.3s ease;
        min-height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0px 8px 18px rgba(111, 78, 55, 0.12);
    }

    .kpi-title {
        font-size: 16px;
        font-weight: 700;
        color: #6F4E37;
        margin-bottom: 10px;
    }

    .kpi-value {
        font-size: 34px;
        font-weight: 800;
        color: #4E342E;
    }

    .kpi-image {
        width: 42px;
        height: 42px;
        margin-bottom: 14px;
        object-fit: contain;
    }

    .chart-card {
        background-color: #FFFDF9;
        border: 1px solid #E5D7C8;
        border-radius: 22px;
        padding: 18px;
        min-height: 560px;
        box-shadow:
            0 0 8px rgba(232, 155, 91, 0.18),
            0 0 18px rgba(181, 101, 118, 0.12),
            0 0 28px rgba(127, 59, 59, 0.08);
    }

    h3 {
        font-size: 24px !important;
        text-align: center !important;
        color: #4E342E !important;
        font-weight: 700 !important;
        margin-bottom: 20px !important;
    }

    /* =========================================
    FORM SUBMIT BUTTON — matches navigation
    ========================================= */
    div[data-testid="stFormSubmitButton"] > button {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 10px !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        padding: 12px 16px !important;
        background-color: #f5e8e1 !important;
        color: #4E342E !important;
        border-radius: 14px !important;
        border: 1px solid #D6B89E !important;
        box-shadow: 0px 3px 8px rgba(111, 78, 55, 0.05) !important;
        transition: all 0.25s ease !important;
        min-height: 52px !important;
        width: 100% !important;
    }

    div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #E7D3C1 !important;
        border-color: #C8A97E !important;
        color: #4E342E !important;
        transform: translateY(-1px) !important;
        box-shadow: 0px 5px 12px rgba(111, 78, 55, 0.10) !important;
    }

    div[data-testid="stFormSubmitButton"] > button:active {
        background-color: #A47551 !important;
        border-color: #A47551 !important;
        color: white !important;
        box-shadow: 0px 6px 16px rgba(111, 78, 55, 0.16) !important;
    }


    </style>
    """,
    unsafe_allow_html=True
)

# =========================================
# LOAD DATA AND MODELS 
# =========================================
@st.cache_data
def load_data():
    df = pd.read_csv("data_cleaned.csv")
    question_categories = pd.read_csv("question_categories.csv")
    return df, question_categories

df, question_categories = load_data()

@st.cache_resource
def load_models():
    model    = joblib.load('models/model_multi.pkl')
    encoders = {
        'depression': joblib.load('models/encoder_depression.pkl'),
        'anxiety':    joblib.load('models/encoder_anxiety.pkl'),
        'stress':     joblib.load('models/encoder_stress.pkl'),
    }
    features = joblib.load('models/features_multi.pkl')
    return model, encoders, features

ml_model, ml_encoders, ml_features = load_models()

# =========================================
# LOAD ICONS
# =========================================
def get_base64(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode()

base64_participants  = get_base64("participants.png")
base64_depression    = get_base64("depression.png")
base64_stress        = get_base64("stress.png")
base64_anxiety       = get_base64("anxiety.png")
base64_risk          = get_base64("risk.png")
base64_mental_health = get_base64("mental_health.png")
base64_predict       = get_base64("predict.png")
base64_high_risk     = get_base64("high_risk.png")
base64_yellow_card   = get_base64("yellow_card.png")
base64_good          = get_base64("good.png")
base64_caution       = get_base64("caution.png")

# ================================
# APP TITLE
# ================================
st.markdown(
    """<div style="
        background-color:#FFFDF9;
        padding:15px;
        border-radius:30px;
        border:1px solid #E5D7C8;
        box-shadow:0px 6px 18px rgba(111, 78, 55, 0.08);
        margin-bottom:30px;
        margin-top:30px;
        text-align:center;">
        <h1 style="color:#4E342E;font-size:48px;font-weight:800;margin-bottom:15px;">
        Mental Health EDA Dashboard
        </h1>
        <p style="color:#6F4E37;font-size:16px;line-height:1.7;margin-top:15px;">
        Exploratory analysis of Depression, Anxiety, and Stress patterns
        using the DASS psychological assessment dataset.
        </p>
    </div>""",
    unsafe_allow_html=True
)

# =========================================
# QUESTIONS DEFINITIONS
# =========================================
depression_questions = {
    'Q3A':  'Unable to feel positive emotion',
    'Q5A':  'Difficulty getting started',
    'Q10A': 'Nothing to look forward to',
    'Q13A': 'Sad and depressed',
    'Q16A': 'Loss of interest',
    'Q17A': 'Feeling worthless',
    'Q21A': 'Life not worthwhile',
    'Q24A': 'No enjoyment',
    'Q26A': 'Down-hearted and blue',
    'Q31A': 'Unable to become enthusiastic',
    'Q34A': 'Feeling pretty worthless',
    'Q37A': 'No meaning in life',
    'Q38A': 'Feeling life was meaningless',
    'Q42A': 'Difficulty working up initiative'
}

anxiety_questions = {
    'Q2A':  'Dryness of mouth',
    'Q4A':  'Breathing difficulty',
    'Q7A':  'Shakiness',
    'Q9A':  'Situations made anxious',
    'Q15A': 'Feeling faint',
    'Q19A': 'Perspiring without reason',
    'Q20A': 'Scared without reason',
    'Q23A': 'Difficulty swallowing',
    'Q25A': 'Heart awareness',
    'Q28A': 'Close to panic',
    'Q30A': 'Fear of being hindered by anxiety',
    'Q36A': 'Feeling terrified',
    'Q40A': 'Worried about panic',
    'Q41A': 'Trembling'
}

stress_questions = {
    'Q1A':  'Upset by trivial things',
    'Q6A':  'Overreacting to situations',
    'Q8A':  'Difficulty relaxing',
    'Q11A': 'Easily upset',
    'Q12A': 'Using nervous energy',
    'Q14A': 'Impatient when delayed',
    'Q18A': 'Touchy or irritable',
    'Q22A': 'Hard to wind down',
    'Q27A': 'Very irritable',
    'Q29A': 'Hard to calm down',
    'Q32A': 'Difficulty tolerating interruptions',
    'Q33A': 'Nervous tension',
    'Q35A': 'Intolerant of delays',
    'Q39A': 'Easily agitated'
}

# =========================================
# SHARED HELPERS
# =========================================
WARM_SCALE = [
    [0.0,  "#F8EDE3"],
    [0.25, "#DDB892"],
    [0.5,  "#E89B5B"],
    [0.75, "#B56576"],
    [1.0,  "#7F3B3B"]
]

def base_layout(fig, height=380, left_margin=20):
    fig.update_layout(
        height=height,
        margin=dict(l=left_margin, r=40, t=60, b=40),
        coloraxis_showscale=False,
        paper_bgcolor="#F7F3ED",
        plot_bgcolor="#F7F3ED",
        legend=dict(orientation="v", y=0.95, x=1.02),
        font=dict(color="#3E2C23"),
        xaxis=dict(
            tickfont=dict(color="#4E342E", size=14),
            title_font=dict(color="#4E342E", size=15)
        ),
        yaxis=dict(
            tickfont=dict(color="#4E342E", size=14),
            title_font=dict(color="#4E342E", size=15)
        ),
        coloraxis_colorbar=dict(
            title_font=dict(color="#4E342E", size=15),
            tickfont=dict(color="#4E342E", size=13)
        )
    )
    return fig

# =========================================
# SIDEBAR
# =========================================
with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align:center;padding-top:10px;padding-bottom:25px;">
            <img src="data:image/png;base64,{base64_mental_health}"
                style="width:80px;height:80px;object-fit:contain;margin-bottom:8px;">
            <div style="color:#4E342E;font-size:24px;font-weight:800;line-height:1.3;">
                Mental Health<br>Analysis Dashboard
            </div>
            <div style="color:#8B5E3C;font-size:13px;margin-top:10px;">
                AI-Powered Psychological Insights
            </div>
            <hr style="margin-top:20px;margin-bottom:10px;border:0.5px solid #DDB892;">
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """<p style="color:#6F4E37;font-size:15px;font-weight:700;
        margin-bottom:5px;letter-spacing:1px;">NAVIGATION</p>""",
        unsafe_allow_html=True
    )

    section = st.radio(
        "",
        [
            "ML Prediction",
            "Overview",
            "Demographics",
            "Psychological Insights",
            "High-Risk Insights"
        ]
    )


# =========================================
# MAIN CONTENT
# =========================================

# ML PREDICTION
if section == "ML Prediction":

    st.markdown("## ML Prediction")
    st.markdown(
        """<p style="color:#8B5E3C;font-size:15px;margin-top:-10px;margin-bottom:20px;">
        Answer the 42 DASS questions below to get your predicted
        Depression, Anxiety and Stress severity levels.
        </p>""",
        unsafe_allow_html=True
    )

    rating_html = (
        '<div style="background-color:#FFFDF9;border:1px solid #E5D7C8;'
        'border-radius:18px;padding:18px 24px;margin-bottom:24px;'
        'box-shadow:0px 4px 10px rgba(111,78,55,0.06);">'
        '<p style="color:#6F4E37;font-size:14px;font-weight:700;'
        'margin-bottom:14px;letter-spacing:1px;">RATING SCALE</p>'
        '<div style="display:flex;justify-content:space-between;margin-bottom:6px;">'
        '<span style="font-size:12px;color:#8B5E3C;font-weight:600;">0 — Never</span>'
        '<span style="font-size:12px;color:#8B5E3C;font-weight:600;">1 — Sometimes</span>'
        '<span style="font-size:12px;color:#8B5E3C;font-weight:600;">2 — Often</span>'
        '<span style="font-size:12px;color:#8B5E3C;font-weight:600;">3 — Almost Always</span>'
        '</div>'
        '<div style="height:8px;border-radius:10px;'
        'background:linear-gradient(to right, #F8EDE3, #DDB892, #E89B5B, #7F3B3B);">'
        '</div>'
        '</div>'
    )
    st.markdown(rating_html, unsafe_allow_html=True)

    st.markdown(
        f"""
        <div style="
            background-color:#FFFDF9;
            border-left:4px solid #DDB892;
            border-radius:12px;
            padding:12px 18px;
            margin-bottom:20px;
            display:flex;
            align-items:center;
            gap:12px;">
            <img src="data:image/png;base64,{base64_caution}"
                style="width:32px;height:32px;object-fit:contain;flex-shrink:0;">
            <span style="font-size:13px;color:#6F4E37;font-weight:500;">
                This tool is for educational purposes only and is not a clinical diagnosis.
                Model accuracy: Depression 86% | Anxiety 83% | Stress 84%
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form("prediction_form"):

        # ── Step 1: Demographics ──
        st.markdown("### Step 1: Your Information")
        st.markdown("<br>", unsafe_allow_html=True)

        d1, d2, d3, d4, d5 = st.columns(5)
        with d1:
            age = st.number_input("Age", min_value=13, max_value=100, value=25)

        with d2:
            gender_label = st.selectbox("Gender", ["Male", "Female"])
            gender = {"Male": 1, "Female": 2}[gender_label]

        with d3:
            education_label = st.selectbox(
                "Education",
                ["Other / Unknown", "High School", "University",
                "Bachelor Degree", "Graduate Degree"]
            )
            education = {
                "Other / Unknown": 0,
                "High School":     1,
                "University":      2,
                "Bachelor Degree": 3,
                "Graduate Degree": 4
            }[education_label]

        with d4:
            familysize = st.number_input("Family Size", min_value=1, max_value=15, value=4)

        with d5:
            married_label = st.selectbox(
                "Marital Status",
                ["Single", "Married", "Divorced / Widowed", "Other"]
            )
            married = {
                "Other":              0,
                "Single":             1,
                "Married":            2,
                "Divorced / Widowed": 3
            }[married_label]

        st.markdown("---")

        # ── Step 2: Questions ──
        st.markdown("### Step 2: DASS-42 Questions")
        st.markdown("<br>", unsafe_allow_html=True)

        responses = {}

        q_groups = {
            "Depression Questions": (depression_questions, base64_depression),
            "Anxiety Questions":    (anxiety_questions,    base64_anxiety),
            "Stress Questions":     (stress_questions,     base64_stress),
        }

        for group_title, (q_dict, b64_icon) in q_groups.items():
            st.markdown(
                f"""<div style="
                    background-color:#FFFDF9;
                    border-left:4px solid #A47551;
                    border-radius:12px;
                    padding:12px 18px;
                    margin-bottom:16px;
                    display:flex;
                    align-items:center;
                    gap:12px;">
                    <img src="data:image/png;base64,{b64_icon}"
                        style="width:32px;height:32px;object-fit:contain;">
                    <p style="color:#4E342E;font-size:16px;
                    font-weight:700;margin:0;">{group_title}</p>
                </div>""",
                unsafe_allow_html=True
            )
            cols = st.columns(2)
            for i, (qcode, qdesc) in enumerate(q_dict.items()):
                with cols[i % 2]:
                    responses[qcode] = st.select_slider(
                        f"**{qcode}** — {qdesc}",
                        options=[0, 1, 2, 3],
                        value=0
                    )
            st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Inject base64 icon into button via CSS
        st.markdown(
            f"""
            <style>
            div[data-testid="stFormSubmitButton"] > button::before {{
                content: '' !important;
                display: inline-block !important;
                width: 24px !important;
                height: 24px !important;
                background-image: url("data:image/png;base64,{base64_predict}") !important;
                background-size: contain !important;
                background-repeat: no-repeat !important;
                background-position: center !important;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

        submitted = st.form_submit_button(
            "Predict My Mental Health",
            use_container_width=True
        )

    # Results
    if submitted:
        demo_values = {
            'age': age, 'gender': gender, 'education': education,
            'familysize': familysize, 'married': married
        }
        all_responses = {**responses, **demo_values}
        input_df      = pd.DataFrame([all_responses])[ml_features]
        preds         = ml_model.predict(input_df)[0]

        dep_label = ml_encoders['depression'].inverse_transform([preds[0]])[0]
        anx_label = ml_encoders['anxiety'].inverse_transform([preds[1]])[0]
        str_label = ml_encoders['stress'].inverse_transform([preds[2]])[0]

        st.markdown("---")
        st.markdown("### Your Results")
        st.markdown("<br>", unsafe_allow_html=True)

        color_map  = {'Low': '#2E7D32', 'Moderate': '#E89B5B', 'High': '#9D4E4E'}
        icon_map   = {'Low': '🟢',      'Moderate': '🟡',       'High': '🔴'}
        advice_map = {
            'Low':      'Your levels appear healthy. Keep maintaining good habits.',
            'Moderate': 'Some symptoms detected. Consider speaking to someone you trust.',
            'High':     'High levels detected. We recommend consulting a mental health professional.'
        }

        c1, c2, c3 = st.columns(3)
        for col, label, target in zip(
            [c1, c2, c3],
            [dep_label, anx_label, str_label],
            ['Depression', 'Anxiety', 'Stress']
        ):
            with col:
                color  = color_map.get(label, '#A47551')
                icon   = icon_map.get(label, '⚪')
                advice = advice_map.get(label, '')
                html = (
                    '<div class="kpi-card" style="min-height:240px;">'
                    f'<div style="font-size:40px;margin-bottom:10px;">{icon}</div>'
                    f'<div class="kpi-title">{target}</div>'
                    f'<div class="kpi-value" style="color:{color};'
                    f'font-size:28px;margin-bottom:12px;">{label}</div>'
                    f'<div style="color:#6F4E37;font-size:13px;'
                    f'line-height:1.5;">{advice}</div>'
                    '</div>'
                )
                st.markdown(html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        high_count = sum(1 for l in [dep_label, anx_label, str_label] if l == 'High')
        mod_count  = sum(1 for l in [dep_label, anx_label, str_label] if l == 'Moderate')

        if high_count >= 2:
            banner_color = '#9D4E4E'
            banner_msg   = "Multiple high-risk conditions detected. Please seek professional support."
            banner_icon  = base64_high_risk
        elif high_count == 1 or mod_count >= 2:
            banner_color = '#E89B5B'
            banner_msg   = "Some elevated levels detected. Monitor your wellbeing closely."
            banner_icon  = base64_yellow_card
        else:
            banner_color = '#C8A97E'
            banner_msg   = "Overall your mental health indicators appear within healthy range."
            banner_icon  = base64_good

        st.markdown(
            f"""<div style="
                background-color:#FFFDF9;
                border-left:5px solid {banner_color};
                border-radius:14px;
                padding:18px 24px;
                margin-top:10px;
                box-shadow:0px 4px 10px rgba(111,78,55,0.06);
                display:flex;
                align-items:center;
                gap:16px;">
                <img src="data:image/png;base64,{banner_icon}"
                    style="width:40px;height:40px;object-fit:contain;flex-shrink:0;">
                <p style="color:{banner_color};font-size:16px;
                font-weight:700;margin:0;">{banner_msg}</p>
            </div>""",
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Score Breakdown")

        score_data = pd.DataFrame({
            'Condition': ['Depression', 'Anxiety', 'Stress'],
            'Level':     [dep_label, anx_label, str_label],
            'Score': [
                {'Low': 1, 'Moderate': 2, 'High': 3}.get(dep_label, 0),
                {'Low': 1, 'Moderate': 2, 'High': 3}.get(anx_label, 0),
                {'Low': 1, 'Moderate': 2, 'High': 3}.get(str_label, 0),
            ]
        })

        fig = px.bar(
            score_data, x='Condition', y='Score',
            color='Condition', text='Level',
            color_discrete_sequence=['#9D4E4E', '#A47551', '#B08968'],
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            height=350,
            yaxis=dict(tickvals=[1,2,3], ticktext=['Low','Moderate','High'], range=[0,4]),
            showlegend=False,
            paper_bgcolor='#F7F3ED', plot_bgcolor='#F7F3ED',
            font=dict(color='#3E2C23'),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(base_layout(fig), use_container_width=True)


# ── OVERVIEW ──────────────────────────────
elif section == "Overview":

    st.markdown("## Overview")

    total_participants = df.shape[0]
    avg_depression     = df["depression_score"].mean()
    avg_anxiety        = df["anxiety_score"].mean()
    avg_stress         = df["stress_score"].mean()

    multi_severe_count = df[
        (df["depression_level"] == "High") &
        (df["anxiety_level"]    == "High") &
        (df["stress_level"]     == "High")
    ].shape[0]

    col1, col2, col3, col4, col5 = st.columns(5)
    kpi_data = [
        (col1, base64_participants, "Total Participants", f"{total_participants:,}"),
        (col2, base64_depression,   "Avg Depression",     f"{avg_depression:.2f}"),
        (col3, base64_stress,       "Avg Stress",         f"{avg_stress:.2f}"),
        (col4, base64_anxiety,      "Avg Anxiety",        f"{avg_anxiety:.2f}"),
        (col5, base64_risk,         "Multi-Severe Cases", f"{multi_severe_count:,}"),
    ]
    for col, b64, title, value in kpi_data:
        with col:
            html = (
                '<div class="kpi-card">'
                f'<img src="data:image/png;base64,{b64}" class="kpi-image">'
                f'<div class="kpi-title">{title}</div>'
                f'<div class="kpi-value">{value}</div>'
                '</div>'
            )
            st.markdown(html, unsafe_allow_html=True)

    st.markdown("---")

    left_col, right_col = st.columns(2, vertical_alignment="top")

    with left_col:
        st.subheader("Mental Health Level Distribution")
        selected_level = st.selectbox(
            "Choose mental health dimension",
            ["Depression Level", "Anxiety Level", "Stress Level"]
        )
        level_mapping = {
            "Depression Level": "depression_level",
            "Anxiety Level":    "anxiety_level",
            "Stress Level":     "stress_level"
        }
        selected_level = level_mapping[selected_level]
        fig = px.pie(
            df, names=selected_level, hole=0.4,
            color_discrete_sequence=["#DDB892", "#E89B5B", "#7F3B3B"]
        )
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(
            height=380, margin=dict(l=20, r=20, t=60, b=20),
            paper_bgcolor="#F7F3ED", plot_bgcolor="#F7F3ED",
            legend=dict(orientation="v", y=0.95, x=1.02),
            font=dict(color="#3E2C23")
        )
        st.plotly_chart(fig, use_container_width=True)

    with right_col:
        st.subheader("Average Severity Across Mental Health")
        st.markdown("<div style='margin-top:84px;'></div>", unsafe_allow_html=True)
        mental_scores = pd.DataFrame({
            'Condition':     ['Depression', 'Anxiety', 'Stress'],
            'Average Score': [
                df['depression_score'].mean(),
                df['anxiety_score'].mean(),
                df['stress_score'].mean()
            ]
        }).sort_values(by='Average Score', ascending=False)
        fig = px.pie(
            mental_scores, values='Average Score', names='Condition',
            color_discrete_sequence=["#E89B5B", "#B56576", "#7F3B3B"]
        )
        fig.update_traces(textinfo='percent+label')
        fig.update_layout(
            height=380, margin=dict(l=20, r=20, t=60, b=20),
            paper_bgcolor="#F7F3ED", plot_bgcolor="#F7F3ED",
            legend=dict(orientation="v", y=0.95, x=1.02),
            font=dict(color="#3E2C23")
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    left_col, right_col = st.columns(2, vertical_alignment="top")

    with left_col:
        st.subheader("Correlation Matrix of Mental Health Scores")
        mental_corr = df[['depression_score', 'anxiety_score', 'stress_score']].corr()
        mental_corr.columns = ['Depression', 'Anxiety', 'Stress']
        mental_corr.index   = ['Depression', 'Anxiety', 'Stress']
        fig = px.imshow(
            mental_corr, text_auto=True,
            color_continuous_scale=WARM_SCALE,
            labels=dict(color='Correlation'),
        )
        st.plotly_chart(base_layout(fig), use_container_width=True)

    with right_col:
        st.subheader("Number of High-Risk Participants by Condition")
        high_risk_counts = pd.DataFrame({
            'Condition': ['Depression', 'Anxiety', 'Stress'],
            'High Risk Count': [
                (df['depression_level'] == 'High').sum(),
                (df['anxiety_level']    == 'High').sum(),
                (df['stress_level']     == 'High').sum()
            ]
        })
        fig = px.bar(
            high_risk_counts, x='Condition', y='High Risk Count',
            color='High Risk Count', text_auto=True,
            color_continuous_scale=WARM_SCALE,
        )
        st.plotly_chart(base_layout(fig), use_container_width=True)


# ── DEMOGRAPHICS ──────────────────────────
elif section == "Demographics":

    st.markdown("## Demographics")
    st.markdown("---")

    selected_level = st.selectbox(
        "Choose mental health dimension",
        ["Depression Score", "Anxiety Score", "Stress Score"]
    )
    level_mapping = {
        "Depression Score": "depression_score",
        "Anxiety Score":    "anxiety_score",
        "Stress Score":     "stress_score"
    }
    selected_level = level_mapping[selected_level]

    left_col, right_col = st.columns(2, vertical_alignment="top")

    with left_col:
        st.subheader('Average Mental Health Scores Across Age Groups')
        age_metric = (
            df.groupby('age_group')[selected_level]
            .mean().reset_index()
            .sort_values(by=selected_level, ascending=False)
        )
        fig = px.bar(
            age_metric, x='age_group', y=selected_level,
            text_auto=True, color=selected_level,
            color_continuous_scale=WARM_SCALE,
        )
        st.plotly_chart(base_layout(fig), use_container_width=True)

    with right_col:
        st.subheader('Average Mental Health Scores Across Genders')
        gender_metric = (
            df.groupby('gender_label')[selected_level]
            .mean().reset_index()
            .sort_values(by=selected_level, ascending=False)
        )
        fig = px.bar(
            gender_metric, x='gender_label', y=selected_level,
            color=selected_level, color_continuous_scale=WARM_SCALE,
            labels={'gender_label': 'Gender',
                    selected_level: f'Average {selected_level.replace("_"," ").title()}'},
            text_auto=True,
        )
        st.plotly_chart(base_layout(fig), use_container_width=True)

    st.markdown("---")

    left_col, right_col = st.columns(2, vertical_alignment="top")

    with left_col:
        st.subheader('Average Mental Health Scores Across Education Degrees')
        Education_mental = df.groupby('education_label')[selected_level].mean().reset_index()
        fig = px.bar(
            Education_mental, x='education_label', y=selected_level,
            color=selected_level, color_continuous_scale=WARM_SCALE,
            labels={'education_label': 'Education Level',
                    selected_level: f'Average {selected_level.replace("_"," ").title()}'},
            text_auto=True,
        )
        st.plotly_chart(base_layout(fig), use_container_width=True)

    with right_col:
        st.subheader('Mental Health Severity Distribution Across Education Levels')
        education_stress = df.groupby('education_label')[selected_level].mean().reset_index()
        fig = px.pie(
            education_stress, names='education_label', values=selected_level,
            color_discrete_sequence=["#F8EDE3", "#DDB892", "#E89B5B", "#B56576", "#7F3B3B"]
        )
        fig.update_traces(textinfo='percent+label')
        fig.update_layout(
            height=380, margin=dict(l=20, r=20, t=60, b=20),
            paper_bgcolor="#F7F3ED", plot_bgcolor="#F7F3ED",
            legend=dict(orientation="v", y=0.95, x=1.02),
            font=dict(color="#3E2C23")
        )
        st.plotly_chart(fig, use_container_width=True)


# ── PSYCHOLOGICAL INSIGHTS ────────────────
elif section == "Psychological Insights":

    st.markdown("## Psychological Insights")
    st.markdown("---")

    st.subheader('Strongest Stress Indicators with High Severity')
    stress_means = (
        df[list(stress_questions.keys())]
        .mean().sort_values(ascending=True).reset_index()
    )
    stress_means.columns = ['Question', 'Average Score']
    stress_means = stress_means.merge(question_categories, on='Question', how='left')
    fig = px.bar(
        stress_means.head(15), x='Average Score', y='Symptom',
        orientation='h', color='Average Score',
        text_auto='.2f', color_continuous_scale=WARM_SCALE,
        labels={'Average Score': 'Average Severity Score', 'Symptom': 'Stress Symptom'},
    )
    st.plotly_chart(base_layout(fig, height=650, left_margin=180), use_container_width=True)

    st.markdown("---")

    st.subheader('Strongest Depression Indicators with High Severity')
    depression_means = (
        df[list(depression_questions.keys())]
        .mean().sort_values(ascending=True).reset_index()
    )
    depression_means.columns = ['Question', 'Average Score']
    depression_means = depression_means.merge(question_categories, on='Question', how='left')
    fig = px.bar(
        depression_means, x='Average Score', y='Symptom',
        orientation='h', color='Average Score',
        text_auto='.2f', color_continuous_scale=WARM_SCALE,
        labels={'Average Score': 'Average Severity Score', 'Symptom': 'Depression Symptom'},
    )
    st.plotly_chart(base_layout(fig, height=650, left_margin=180), use_container_width=True)

    st.markdown("---")

    st.subheader('Strongest Anxiety Indicators with High Severity')
    anxiety_means = (
        df[list(anxiety_questions.keys())]
        .mean().sort_values(ascending=True).reset_index()
    )
    anxiety_means.columns = ['Question', 'Average Score']
    anxiety_means = anxiety_means.merge(question_categories, on='Question', how='left')
    fig = px.bar(
        anxiety_means, x='Average Score', y='Symptom',
        orientation='h', color='Average Score',
        text_auto='.2f', color_continuous_scale=WARM_SCALE,
        labels={'Average Score': 'Average Severity Score', 'Symptom': 'Anxiety Symptom'},
    )
    st.plotly_chart(base_layout(fig, height=650, left_margin=180), use_container_width=True)


# ── HIGH-RISK INSIGHTS ────────────────────
elif section == "High-Risk Insights":

    st.markdown("## High-Risk Insights")
    st.markdown("---")

    left_col, right_col = st.columns(2, vertical_alignment="top")

    with left_col:
        st.subheader("High-Risk Participants by Condition")
        high_risk_counts = pd.DataFrame({
            "Condition": ["Depression", "Anxiety", "Stress"],
            "High Risk Count": [
                (df["depression_level"] == "High").sum(),
                (df["anxiety_level"]    == "High").sum(),
                (df["stress_level"]     == "High").sum()
            ]
        })
        fig = px.bar(
            high_risk_counts, x="Condition", y="High Risk Count",
            color="High Risk Count", text_auto=True,
            color_continuous_scale=WARM_SCALE,
        )
        st.plotly_chart(base_layout(fig), use_container_width=True)

    with right_col:
        st.subheader("Multiple Severe Conditions")
        multiple_severe = df[
            (df["depression_level"] == "High") &
            (df["anxiety_level"]    == "High") &
            (df["stress_level"]     == "High")
        ]
        severity_combo = pd.DataFrame({
            "Category": ["Multiple Severe Conditions", "Other Participants"],
            "Count": [
                multiple_severe.shape[0],
                df.shape[0] - multiple_severe.shape[0]
            ]
        })
        fig = px.pie(
            severity_combo, names="Category", values="Count",
            hole=0.4, color_discrete_sequence=["#DDB892", "#7F3B3B"]
        )
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(
            height=380, margin=dict(l=20, r=20, t=60, b=20),
            paper_bgcolor="#F7F3ED", plot_bgcolor="#F7F3ED",
            legend=dict(orientation="v", y=0.95, x=1.02),
            font=dict(color="#3E2C23")
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    high_risk_df = df[
        (df["depression_level"] == "High") |
        (df["anxiety_level"]    == "High") |
        (df["stress_level"]     == "High")
    ]

    left_col, right_col = st.columns(2, vertical_alignment="top")

    with left_col:
        st.subheader("High-Risk Participants by Gender")
        gender_risk = (
            high_risk_df.groupby("gender_label")
            .size().reset_index(name="High Risk Count")
            .sort_values(by="High Risk Count", ascending=False)
        )
        fig = px.bar(
            gender_risk, x="gender_label", y="High Risk Count",
            color="High Risk Count", text_auto=True,
            color_continuous_scale=WARM_SCALE,
            labels={"gender_label": "Gender", "High Risk Count": "High Risk Participants"}
        )
        st.plotly_chart(base_layout(fig), use_container_width=True)

    with right_col:
        st.subheader("High-Risk Participants by Age Group")
        age_risk = (
            high_risk_df.groupby("age_group")
            .size().reset_index(name="High Risk Count")
            .sort_values(by="High Risk Count", ascending=False)
        )
        fig = px.bar(
            age_risk, x="age_group", y="High Risk Count",
            color="High Risk Count", text_auto=True,
            color_continuous_scale=WARM_SCALE,
            labels={"age_group": "Age Group", "High Risk Count": "High Risk Participants"}
        )
        st.plotly_chart(base_layout(fig), use_container_width=True)

st.markdown(
    """
    <div style="
        text-align:center;
        margin-top:60px;
        padding:20px;
        border-top:1px solid #E5D7C8;">
        <p style="
            color:#8B5E3C;
            font-size:14px;
            margin-bottom:10px;
            letter-spacing:0.5px;">
            Developed by
            <span style="
                color:#A47551;
                font-weight:800;
                font-size:15px;">
                Alaa Mekawi
            </span>
        </p>
        <div style="display:flex;justify-content:center;gap:12px;">
            <a href="https://www.linkedin.com/in/alaa-mekawi-37b245221/"
               target="_blank"
               style="
                   color:#FFFDF9;
                   text-decoration:none;
                   font-size:12px;
                   font-weight:600;
                   background-color:#A47551;
                   padding:6px 18px;
                   border-radius:20px;
                   border:1px solid #8B5E3C;
                   letter-spacing:0.5px;">
                LinkedIn
            </a>
            <a href="https://github.com/ALAAMEKAWY56"
               target="_blank"
               style="
                   color:#A47551;
                   text-decoration:none;
                   font-size:12px;
                   font-weight:600;
                   background-color:#FFFDF9;
                   padding:6px 18px;
                   border-radius:20px;
                   border:1px solid #A47551;
                   letter-spacing:0.5px;">
                GitHub
            </a>
        </div>
        <p style="
            color:#C8A97E;
            font-size:11px;
            margin-top:10px;
            margin-bottom:0;
            letter-spacing:1px;">
            &copy; 2026 &nbsp;&#183;&nbsp; DASS-42 Mental Health Analysis Dashboard
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
