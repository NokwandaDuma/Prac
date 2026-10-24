"""
AI Job Loss Risk - Decision-Support Dashboard (Task 1: Predicting Job Loss Risk)

Run:  streamlit run app.py
Needs (same folder):
    ai_job_loss_risk_clean.csv          <- saved by the notebook
    random_forest_job_risk_model.pkl    <- saved by the notebook
"""
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

st.set_page_config(page_title="AI Job Loss Risk - South Africa", layout="wide")

EDU_ORDER = ["No schooling/Primary", "Some secondary", "Matric",
             "Certificate/Diploma", "Degree", "Postgraduate"]
SKILL_ORDER = ["Low", "Medium", "High"]
AGE_BANDS = ["18-24", "25-34", "35-44", "45-54", "55-64"]


# --------------------------------------------------------------------------
# Load data + model
# --------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("ai_job_loss_risk_clean.csv")
    df["Age_Band"] = pd.cut(df["Age"], bins=[17, 24, 34, 44, 54, 64], labels=AGE_BANDS)
    return df


@st.cache_resource
def load_model():
    return joblib.load("random_forest_job_risk_model.pkl")


def encode(frame, md):
    """One-hot encode and align to the exact training columns."""
    X = pd.get_dummies(frame[md["numeric_features"] + md["categorical_columns"]],
                       columns=md["categorical_columns"], dtype=int)
    return X.reindex(columns=md["features"], fill_value=0)


@st.cache_data
def score_all(_md, df):
    X = encode(df, _md)
    out = df.copy()
    out["Risk_Probability"] = _md["model"].predict_proba(X)[:, 1]
    out["Is_Test_Row"] = out.index.isin(_md["test_index"])
    return out


def grouped_permutation_importance(model, md, frame, y, n_repeats=3, seed=42):
    """Shuffle each ORIGINAL variable (all its dummy columns together) and
    measure the drop in F1. Easier to read than 50+ dummy columns."""
    rng = np.random.default_rng(seed)
    X = encode(frame, md)
    base = f1_score(y, model.predict(X), zero_division=0)
    rows = []
    for var in md["numeric_features"] + md["categorical_columns"]:
        cols = [c for c in X.columns if c == var or c.startswith(var + "_")]
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(Xp))
            Xp[cols] = X[cols].values[perm]
            drops.append(base - f1_score(y, model.predict(Xp), zero_division=0))
        rows.append((var, float(np.mean(drops))))
    return pd.DataFrame(rows, columns=["Variable", "Importance (F1 drop)"]) \
        .sort_values("Importance (F1 drop)", ascending=False)


try:
    raw = load_data()
    md = load_model()
except FileNotFoundError as e:
    st.error(f"Missing file: {e.filename}. Run the notebook first so it saves the "
             "cleaned CSV and the .pkl model into this folder.")
    st.stop()

data = score_all(md, raw)

# --------------------------------------------------------------------------
# Sidebar filters
# --------------------------------------------------------------------------
st.sidebar.title("Filters")
st.sidebar.caption("All views update to the selected group of workers.")

years = sorted(data["Year"].unique())
yr = st.sidebar.slider("Time period", int(min(years)), int(max(years)),
                       (int(min(years)), int(max(years))))
industries = st.sidebar.multiselect("Industry / sector", sorted(data["Industry"].unique()))
provinces = st.sidebar.multiselect("Province", sorted(data["Province"].unique()))
metro = st.sidebar.multiselect("Metro status", sorted(data["Metro_Status"].unique()))
edu = st.sidebar.multiselect("Education level", EDU_ORDER)
age_band = st.sidebar.multiselect("Age band", AGE_BANDS)
skill = st.sidebar.multiselect("Skill category", SKILL_ORDER)
gender = st.sidebar.multiselect("Gender", sorted(data["Gender"].unique()))
st.sidebar.caption("Leave a filter empty to include everyone.")

f = data[(data["Year"] >= yr[0]) & (data["Year"] <= yr[1])]
for col, sel in [("Industry", industries), ("Province", provinces), ("Metro_Status", metro),
                 ("Education", edu), ("Age_Band", age_band), ("Skill_Level", skill),
                 ("Gender", gender)]:
    if sel:
        f = f[f[col].isin(sel)]

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("AI-Driven Job Loss Risk - South Africa")
st.caption("Decision-support tool for policymakers, labour unions and industry stakeholders. "
           "Data: synthetic worker-level dataset structured like StatsSA QLFS with "
           "automation-exposure indicators (for demonstration).")

if len(f) == 0:
    st.warning("No workers match the current filters. Please widen your selection.")
    st.stop()

tab_eda, tab_clf, tab_exp, tab_sim, tab_pol = st.tabs(
    ["EDA", "Classification", "Explainability", "Worker risk calculator", "Policy notes"])

# ==========================================================================
# EDA
# ==========================================================================
with tab_eda:
    st.subheader("Summary for the selected workers")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sample size", f"{len(f):,}")
    c2.metric("High-risk (actual)", f"{f['High_Risk'].mean() * 100:.1f}%")
    c3.metric("Low-risk workers", f"{(f['High_Risk'] == 0).sum():,}")
    c4.metric("Avg automation exposure", f"{f['Automation_Exposure_Index'].mean():.2f}")
    c5.metric("Avg age", f"{f['Age'].mean():.1f}")

    st.markdown("**Class balance and key covariates**")
    cb = f["High_Risk"].map({0: "Low risk", 1: "High risk"}).value_counts().rename("Workers").to_frame()
    cb["Share (%)"] = (cb["Workers"] / len(f) * 100).round(1)
    a, b = st.columns([1, 2])
    a.dataframe(cb, width="stretch")
    covs = ["Age", "Years_Experience", "Digital_Skills_Score", "Monthly_Income_ZAR",
            "Routine_Task_Index", "Industry_AI_Adoption", "Automation_Exposure_Index"]
    b.dataframe(f[covs].describe().T[["mean", "std", "min", "50%", "max"]].round(2),
                width="stretch")
    if len(f) < 100:
        st.info("Small sample: treat percentages below as indicative only.")

    st.divider()
    l, r = st.columns(2)
    by_ind = (f.groupby("Industry")["High_Risk"].agg(["mean", "size"]).reset_index()
              .rename(columns={"mean": "High-risk rate", "size": "Workers"})
              .sort_values("High-risk rate"))
    fig = px.bar(by_ind, x="High-risk rate", y="Industry", orientation="h",
                 hover_data=["Workers"], title="High-risk rate by industry")
    fig.update_xaxes(tickformat=".0%")
    l.plotly_chart(fig, width="stretch")

    by_prov = (f.groupby("Province")["High_Risk"].mean().reset_index()
               .sort_values("High_Risk"))
    fig = px.bar(by_prov, x="High_Risk", y="Province", orientation="h",
                 title="High-risk rate by province", labels={"High_Risk": "High-risk rate"})
    fig.update_xaxes(tickformat=".0%")
    r.plotly_chart(fig, width="stretch")

    l, r = st.columns(2)
    fig = px.violin(f, x="Skill_Level", y="Automation_Exposure_Index", color="Skill_Level",
                    box=True, category_orders={"Skill_Level": SKILL_ORDER},
                    title="Automation exposure by skill level")
    l.plotly_chart(fig, width="stretch")

    by_edu = f.groupby("Education")["High_Risk"].mean().reindex(EDU_ORDER).dropna().reset_index()
    fig = px.bar(by_edu, x="Education", y="High_Risk", title="High-risk rate by education",
                 labels={"High_Risk": "High-risk rate"})
    fig.update_yaxes(tickformat=".0%")
    r.plotly_chart(fig, width="stretch")

    heat = f.pivot_table(index="Industry", columns="Skill_Level", values="High_Risk",
                         aggfunc="mean").reindex(columns=SKILL_ORDER)
    fig = px.imshow(heat, text_auto=".0%", aspect="auto", color_continuous_scale="Reds",
                    title="High-risk rate: industry x skill level")
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Feature distribution by risk class**")
    num_choice = st.selectbox("Choose a feature", covs, index=4)
    fig = px.histogram(f.assign(Risk=f["High_Risk"].map({0: "Low risk", 1: "High risk"})),
                       x=num_choice, color="Risk", barmode="overlay", opacity=0.6, nbins=40)
    st.plotly_chart(fig, width="stretch")

# ==========================================================================
# Classification
# ==========================================================================
with tab_clf:
    st.subheader("Model predictions for the selected workers")
    st.caption("Model: Random Forest (trained on a balanced training set). Because the training "
               "data was oversampled, predicted probabilities run higher than real-world rates - "
               "use them to **rank** workers by risk rather than as exact chances.")

    only_test = st.checkbox("Evaluate on hold-out test workers only (recommended)", value=True,
                            help="The model never saw these workers during training, so "
                                 "metrics are an honest estimate of real-world performance.")
    threshold = st.slider("Decision threshold for 'High risk'", 0.10, 0.90, 0.50, 0.05,
                          help="Lower = catch more at-risk workers but flag more people who are fine.")

    ev = f[f["Is_Test_Row"]] if only_test else f
    f_pred = f.assign(Predicted=(f["Risk_Probability"] >= threshold).astype(int))
    ev = ev.assign(Predicted=(ev["Risk_Probability"] >= threshold).astype(int))

    c1, c2, c3 = st.columns(3)
    c1.metric("Workers flagged high-risk (filtered)", f"{int(f_pred['Predicted'].sum()):,}",
              f"{f_pred['Predicted'].mean() * 100:.1f}% of selection", delta_color="off")
    c2.metric("Average predicted risk", f"{f['Risk_Probability'].mean() * 100:.1f}%")
    c3.metric("Workers used for evaluation", f"{len(ev):,}")

    if len(ev) < 30 or ev["High_Risk"].nunique() < 2:
        st.info("Not enough workers (or only one risk class) in this selection to compute "
                "reliable metrics. Widen the filters.")
    else:
        y_true, y_hat = ev["High_Risk"], ev["Predicted"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{accuracy_score(y_true, y_hat):.2f}")
        m2.metric("Precision", f"{precision_score(y_true, y_hat, zero_division=0):.2f}")
        m3.metric("Recall", f"{recall_score(y_true, y_hat, zero_division=0):.2f}")
        m4.metric("F1", f"{f1_score(y_true, y_hat, zero_division=0):.2f}")

        cm = confusion_matrix(y_true, y_hat, labels=[0, 1])
        l, r = st.columns(2)
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        x=["Pred: Low", "Pred: High"], y=["Actual: Low", "Actual: High"],
                        title="Confusion matrix")
        l.plotly_chart(fig, width="stretch")
        fig = px.histogram(ev.assign(Actual=ev["High_Risk"].map({0: "Low risk", 1: "High risk"})),
                           x="Risk_Probability", color="Actual", nbins=30, barmode="overlay",
                           opacity=0.65, title="Predicted risk probability")
        fig.add_vline(x=threshold, line_dash="dash")
        r.plotly_chart(fig, width="stretch")

    st.markdown(
        "**How to read the trade-off.** *Recall* is the share of truly at-risk workers the model "
        "catches; *precision* is the share of flagged workers who really are at risk. For an "
        "**early-intervention or re-skilling programme**, missing an at-risk worker costs more than "
        "offering training to someone who would have been fine, so favour **higher recall** (lower "
        "the threshold). When support is **scarce or expensive**, favour **higher precision** "
        "(raise the threshold).")

    st.markdown("**Highest-risk workers in the selection**")
    show = ["Industry", "Province", "Education", "Skill_Level", "Age", "Employment_Type",
            "Automation_Exposure_Index", "Risk_Probability", "High_Risk"]
    top = f_pred.sort_values("Risk_Probability", ascending=False)[show].head(50)
    st.dataframe(top.style.format({"Risk_Probability": "{:.0%}"}), width="stretch")
    st.download_button("Download full filtered predictions (CSV)",
                       f_pred[show].to_csv(index=False).encode(), "filtered_predictions.csv")

# ==========================================================================
# Explainability
# ==========================================================================
with tab_exp:
    st.subheader("What drives risk for the selected workers?")
    base = f[f["Is_Test_Row"]]
    if len(base) < 50 or base["High_Risk"].nunique() < 2:
        base = f
        st.caption("Few hold-out workers in this selection, so all selected workers are used.")
    if len(base) > 1500:
        base = base.sample(1500, random_state=42)
    if len(base) < 30 or base["High_Risk"].nunique() < 2:
        st.info("Not enough data under the current filters. Widen the selection.")
    else:
        with st.spinner("Calculating permutation importance..."):
            imp = grouped_permutation_importance(md["model"], md, base, base["High_Risk"])
        top = imp.head(10).sort_values("Importance (F1 drop)")
        fig = px.bar(top, x="Importance (F1 drop)", y="Variable", orientation="h",
                     title="Permutation importance (how much accuracy is lost if a variable is scrambled)")
        st.plotly_chart(fig, width="stretch")
        lead = imp.iloc[0]["Variable"].replace("_", " ")
        st.markdown(f"Under the current filters, **{lead}** has the strongest influence on predicted "
                    "risk. Variables near zero add little once the others are known.")
        st.caption("Importance is measured on the combined variable (e.g. all Industry categories "
                   "together). Importance shows association in the data, not proof of cause.")

# ==========================================================================
# Individual worker calculator
# ==========================================================================
with tab_sim:
    st.subheader("Individual worker risk calculator")
    st.caption("Describe a worker to see their predicted risk of AI-driven job loss.")
    ind_df = data.groupby("Industry")[["Routine_Task_Index", "Industry_AI_Adoption"]].mean()
    c1, c2, c3 = st.columns(3)
    w_ind = c1.selectbox("Industry", sorted(data["Industry"].unique()))
    w_prov = c1.selectbox("Province", sorted(data["Province"].unique()))
    w_metro = c1.selectbox("Metro status", sorted(data["Metro_Status"].unique()))
    w_edu = c2.selectbox("Education", EDU_ORDER, index=2)
    w_skill = c2.selectbox("Skill level", SKILL_ORDER)
    w_emp = c2.selectbox("Employment type", ["Permanent", "Contract", "Informal"])
    w_age = c3.slider("Age", 18, 64, 40)
    w_exp = c3.slider("Years of experience", 0, 45, min(15, max(0, w_age - 20)))
    w_dig = c3.slider("Digital skills (1-5)", 1, 5, 3)
    c1, c2, c3 = st.columns(3)
    w_gender = c1.selectbox("Gender", ["Female", "Male"])
    w_union = c1.selectbox("Union member", ["No", "Yes"])
    w_firm = c2.selectbox("Firm size", ["1-9", "10-49", "50-249", "250+"])
    w_train = c2.selectbox("Training in last 2 years", ["No", "Yes"])
    w_inc = c3.number_input("Monthly income (ZAR)", 0, 100000, 12000, step=500)
    d_routine = float(ind_df.loc[w_ind, "Routine_Task_Index"])
    d_ai = float(ind_df.loc[w_ind, "Industry_AI_Adoption"])
    w_routine = st.slider("Routine task index (0 = creative/varied, 1 = fully routine)", 0.0, 1.0,
                          round(d_routine, 2), 0.01,
                          help=f"Defaults to the {w_ind} average.")
    w_ai = st.slider("Industry AI adoption (0-100)", 0, 100, int(d_ai))
    w_expo = st.slider("Automation exposure index", 0.0, 1.0,
                       float(round(min(0.99, 0.75 * w_routine + 0.20 * w_ai / 100), 2)), 0.01)

    worker = pd.DataFrame([{
        "Age": w_age, "Years_Experience": w_exp, "Digital_Skills_Score": w_dig,
        "Monthly_Income_ZAR": w_inc, "Routine_Task_Index": w_routine,
        "Industry_AI_Adoption": w_ai, "Automation_Exposure_Index": w_expo,
        "Province": w_prov, "Metro_Status": w_metro, "Industry": w_ind, "Gender": w_gender,
        "Education": w_edu, "Skill_Level": w_skill, "Employment_Type": w_emp,
        "Union_Member": w_union, "Firm_Size": w_firm, "Recent_Training": w_train}])
    p = float(md["model"].predict_proba(encode(worker, md))[0, 1])
    label = "HIGH RISK" if p >= 0.5 else ("ELEVATED" if p >= 0.35 else "LOW RISK")
    st.metric("Predicted probability of high AI-driven job-loss risk", f"{p * 100:.1f}%", label,
              delta_color="off")
    st.progress(min(max(p, 0.0), 1.0))
    avg = float(data["Risk_Probability"].mean())
    st.caption(f"For comparison, the average worker in the dataset scores {avg * 100:.1f}%. "
               "Scores are best used to rank workers, not as exact probabilities.")

# ==========================================================================
# Policy notes (auto-generated from the current filters)
# ==========================================================================
with tab_pol:
    st.subheader("Policy notes")
    st.caption("Plain-language summary that updates with your filters.")
    overall = f["High_Risk"].mean()
    st.markdown(f"**Overall.** In the selected group of **{len(f):,} workers**, about "
                f"**{overall * 100:.0f}%** are at high risk of losing their job to AI-driven "
                f"automation.")

    def top_group(col, n=1, min_n=30):
        g = f.groupby(col)["High_Risk"].agg(["mean", "size"])
        g = g[g["size"] >= min_n].sort_values("mean", ascending=False)
        return g.head(n)

    notes = []
    g = top_group("Industry", 3)
    if len(g):
        txt = ", ".join(f"{i} ({r['mean'] * 100:.0f}%)" for i, r in g.iterrows())
        notes.append(f"**Most exposed sectors:** {txt}. Prioritise these for transition planning "
                     "and employer-union engagement.")
    g = top_group("Skill_Level")
    if len(g):
        i, r = g.index[0], g.iloc[0]
        notes.append(f"**Skill gap:** {i}-skill workers carry the highest risk ({r['mean'] * 100:.0f}%). "
                     "Target re-skilling funds (digital literacy, data-entry and customer-service "
                     "automation) at this group first.")
    g = top_group("Employment_Type")
    if len(g):
        i, r = g.index[0], g.iloc[0]
        notes.append(f"**Job security:** {i} workers are most at risk ({r['mean'] * 100:.0f}%); they "
                     "are least likely to receive employer-funded training or retrenchment protection.")
    g = top_group("Age_Band")
    if len(g):
        i, r = g.index[0], g.iloc[0]
        notes.append(f"**Age:** the {i} age band shows the highest risk ({r['mean'] * 100:.0f}%).")
    g = top_group("Province")
    if len(g):
        i, r = g.index[0], g.iloc[0]
        notes.append(f"**Geography:** {i} has the highest risk rate in this selection "
                     f"({r['mean'] * 100:.0f}%).")
    tr = f.groupby("Recent_Training")["High_Risk"].mean()
    if {"Yes", "No"} <= set(tr.index):
        notes.append(f"**Training matters:** workers with recent training have a "
                     f"{tr['Yes'] * 100:.0f}% high-risk rate versus {tr['No'] * 100:.0f}% without "
                     "- supporting wider access to short skills programmes.")
    for n in notes:
        st.markdown("- " + n)

    st.markdown("**Suggested actions**")
    st.markdown("- Use the *Classification* tab with a **lower threshold** to build early-warning "
                "lists for re-skilling outreach.\n"
                "- Combine model results with local labour-market knowledge before acting on "
                "individual cases.\n"
                "- Re-run this analysis each year as AI adoption changes.")
    st.warning("Limitations: the dataset is synthetic and for demonstration; predictions show "
               "statistical association, not certainty about any individual's future.")
