import os

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "ai_student_impact_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "rf_model.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "model_columns.pkl")

CATEGORICAL_PREFIXES = [
    "Major_Category",
    "Year_of_Study",
    "Primary_Use_Case",
    "Prompt_Engineering_Skill",
    "Institutional_Policy",
    "Burnout_Risk_Level",
]
NUMERIC_FEATURES = [
    "Weekly_GenAI_Hours",
    "Tool_Diversity",
    "Traditional_Study_Hours",
    "Perceived_AI_Dependency",
    "Anxiety_Level_During_Exams",
    "Skill_Retention_Score",
]

st.set_page_config(
    page_title="AI Impact on Student GPA",
    page_icon="🎓",
    layout="wide",
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    text_cols = df.select_dtypes(include="object").columns
    for c in text_cols:
        df[c] = df[c].astype(str).str.strip()

    df = df.drop_duplicates()
    df = df.drop_duplicates(subset="Student_ID", keep="first")
    df["Improved"] = (df["Post_Semester_GPA"] > df["Pre_Semester_GPA"]).astype(int)
    return df


@st.cache_data
def build_features(df):
    drop_cols = ["Student_ID", "Pre_Semester_GPA", "Post_Semester_GPA", "Improved"]
    X = df.drop(columns=drop_cols)
    X = pd.get_dummies(X, drop_first=True)
    y = df["Improved"]
    return X, y


@st.cache_resource
def load_or_train_model(df):
    X, y = build_features(df)

    if os.path.exists(MODEL_PATH) and os.path.exists(COLUMNS_PATH):
        model = joblib.load(MODEL_PATH)
        model_columns = joblib.load(COLUMNS_PATH)
    else:
        model = RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, class_weight="balanced"
        )
        model.fit(X, y)
        model_columns = X.columns.tolist()
        joblib.dump(model, MODEL_PATH)
        joblib.dump(model_columns, COLUMNS_PATH)

    return model, model_columns


@st.cache_data
def evaluate_model(_model, df, model_columns):
    X, y = build_features(df)
    X = X.reindex(columns=model_columns, fill_value=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    preds = _model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
    }
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    return metrics, cm, report


def build_input_row(inputs, model_columns):
    row = pd.Series(0.0, index=model_columns, dtype=float)

    for col in NUMERIC_FEATURES:
        if col in row.index:
            row[col] = inputs[col]

    row["Paid_Subscription"] = 1.0 if inputs["Paid_Subscription"] else 0.0

    for prefix in CATEGORICAL_PREFIXES:
        col_name = f"{prefix}_{inputs[prefix]}"
        if col_name in row.index:
            row[col_name] = 1.0

    return pd.DataFrame([row])[model_columns]


df = load_data()
model, model_columns = load_or_train_model(df)

st.title("🎓 AI Impact on Student GPA")
st.caption(
    "Exploring how GenAI tool usage relates to semester GPA change, "
    "with a Random Forest classifier predicting whether a student's GPA improved."
)

tab_overview, tab_explore, tab_predict, tab_performance = st.tabs(
    ["Overview", "Explore Data", "Predict", "Model Performance"]
)

with tab_overview:
    st.subheader("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Students", f"{len(df):,}")
    col2.metric("Improved GPA", f"{df['Improved'].mean() * 100:.1f}%")
    col3.metric("Avg Pre-GPA", f"{df['Pre_Semester_GPA'].mean():.2f}")
    col4.metric("Avg Post-GPA", f"{df['Post_Semester_GPA'].mean():.2f}")

    st.divider()
    st.markdown("**Sample rows**")
    st.dataframe(df.head(50), use_container_width=True)

    st.markdown("**Summary statistics (numeric columns)**")
    st.dataframe(df.describe(), use_container_width=True)

with tab_explore:
    st.subheader("Explore the Data")

    left, right = st.columns(2)

    with left:
        st.markdown("**GPA change distribution**")
        df_plot = df.copy()
        df_plot["GPA_Change"] = df_plot["Post_Semester_GPA"] - df_plot["Pre_Semester_GPA"]
        fig = px.histogram(df_plot, x="GPA_Change", nbins=50, color="Improved")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Weekly GenAI hours vs. improvement**")
        fig2 = px.box(df, x="Improved", y="Weekly_GenAI_Hours", color="Improved")
        st.plotly_chart(fig2, use_container_width=True)

    with right:
        st.markdown("**Improvement rate by category**")
        cat_choice = st.selectbox(
            "Group by",
            [
                "Major_Category",
                "Year_of_Study",
                "Primary_Use_Case",
                "Prompt_Engineering_Skill",
                "Institutional_Policy",
                "Burnout_Risk_Level",
            ],
        )
        grouped = df.groupby(cat_choice)["Improved"].mean().sort_values(ascending=False)
        fig3 = px.bar(grouped, labels={"value": "Improvement Rate", cat_choice: cat_choice})
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("**Traditional study hours vs. improvement**")
        fig4 = px.box(df, x="Improved", y="Traditional_Study_Hours", color="Improved")
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("**Correlation between numeric features**")
    numeric_df = df.select_dtypes(include="number").drop(columns=["Student_ID"])
    corr = numeric_df.corr()
    fig5 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    st.plotly_chart(fig5, use_container_width=True)

with tab_predict:
    st.subheader("Predict GPA Improvement")
    st.caption("Enter a student's profile to estimate the likelihood their GPA improves.")

    c1, c2, c3 = st.columns(3)

    with c1:
        major = st.selectbox("Major Category", sorted(df["Major_Category"].unique()))
        year = st.selectbox("Year of Study", sorted(df["Year_of_Study"].unique()))
        use_case = st.selectbox("Primary Use Case", sorted(df["Primary_Use_Case"].unique()))
        skill = st.selectbox(
            "Prompt Engineering Skill", sorted(df["Prompt_Engineering_Skill"].unique())
        )

    with c2:
        policy = st.selectbox("Institutional Policy", sorted(df["Institutional_Policy"].unique()))
        burnout = st.selectbox("Burnout Risk Level", sorted(df["Burnout_Risk_Level"].unique()))
        paid_sub = st.checkbox("Paid AI Subscription", value=False)
        tool_diversity = st.slider("Tool Diversity (# tools used)", 1, 5, 2)

    with c3:
        genai_hours = st.slider("Weekly GenAI Hours", 0.0, 40.0, 8.0, 0.5)
        study_hours = st.slider("Traditional Study Hours", 1.0, 35.0, 12.0, 0.5)
        dependency = st.slider("Perceived AI Dependency (1-10)", 1, 10, 5)
        anxiety = st.slider("Anxiety Level During Exams (1-10)", 1, 10, 5)
        retention = st.slider("Skill Retention Score", 10.0, 100.0, 60.0, 1.0)

    inputs = {
        "Major_Category": major,
        "Year_of_Study": year,
        "Primary_Use_Case": use_case,
        "Prompt_Engineering_Skill": skill,
        "Institutional_Policy": policy,
        "Burnout_Risk_Level": burnout,
        "Paid_Subscription": paid_sub,
        "Tool_Diversity": tool_diversity,
        "Weekly_GenAI_Hours": genai_hours,
        "Traditional_Study_Hours": study_hours,
        "Perceived_AI_Dependency": dependency,
        "Anxiety_Level_During_Exams": anxiety,
        "Skill_Retention_Score": retention,
    }

    if st.button("Predict", type="primary"):
        input_row = build_input_row(inputs, model_columns)
        prediction = model.predict(input_row)[0]
        probability = model.predict_proba(input_row)[0][1]

        st.divider()
        if prediction == 1:
            st.success(f"Predicted: GPA likely to **improve** ({probability * 100:.1f}% confidence)")
        else:
            st.error(f"Predicted: GPA **not** likely to improve ({(1 - probability) * 100:.1f}% confidence)")

        st.progress(min(max(probability, 0.0), 1.0))

with tab_performance:
    st.subheader("Model Performance (held-out test set)")

    metrics, cm, report = evaluate_model(model, df, model_columns)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics['accuracy']:.2f}")
    m2.metric("Precision", f"{metrics['precision']:.2f}")
    m3.metric("Recall", f"{metrics['recall']:.2f}")
    m4.metric("F1 Score", f"{metrics['f1']:.2f}")

    left, right = st.columns(2)

    with left:
        st.markdown("**Confusion Matrix**")
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=["Predicted: No Improvement", "Predicted: Improvement"],
            y=["Actual: No Improvement", "Actual: Improvement"],
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with right:
        st.markdown("**Classification Report**")
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(2), use_container_width=True)

    st.markdown("**Top Feature Importances**")
    importance = pd.Series(model.feature_importances_, index=model_columns)
    importance = importance.sort_values(ascending=False).head(15)
    fig_imp = px.bar(importance[::-1], orientation="h", labels={"value": "Importance", "index": ""})
    st.plotly_chart(fig_imp, use_container_width=True)
