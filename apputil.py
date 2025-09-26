
import re
import pandas as pd
import numpy as np
import plotly.express as px

AGE_CATS = pd.CategoricalDtype(
    categories=["Child (<=12)", "Teen (13–19)", "Adult (20–59)", "Senior (60+)"],
    ordered=True
)

def _ensure_core_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce core numeric columns and keep a copy."""
    out = df.copy()
    for col in ["Pclass", "Age", "SibSp", "Parch", "Fare", "Survived"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out

def _add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Define bins to match categories exactly:
    # Child (up to 12) => <=12
    # Teen (13–19)     => 13..19
    # Adult (20–59)    => 20..59
    # Senior (60+)     => >=60
    bins = [-np.inf, 12, 19, 59, np.inf]
    labels = AGE_CATS.categories
    out["age_group"] = pd.cut(out["Age"], bins=bins, labels=labels, include_lowest=True).astype(AGE_CATS)
    return out

def _ordered_pclass(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Pclass" in out.columns:
        out["Pclass"] = pd.Categorical(out["Pclass"], categories=[1, 2, 3], ordered=True)
    return out


# Exercise 1

def survival_demographics(df: pd.DataFrame) -> pd.DataFrame:
    df1 = _ensure_core_columns(df)
    df1 = _add_age_group(df1)
    df1 = _ordered_pclass(df1)

    grouped = (
        df1
        .groupby(["Pclass", "Sex", "age_group"], dropna=False, observed=True, as_index=False)
        .agg(
            n_passengers=("Survived", "size"),
            n_survivors=("Survived", lambda s: np.nansum(s == 1))
        )
    )
    grouped["survival_rate"] = grouped["n_survivors"] / grouped["n_passengers"]
    grouped = grouped.sort_values(["Pclass", "Sex", "age_group"]).reset_index(drop=True)
    return grouped


def visualize_demographic(df: pd.DataFrame):
    """
    Plotly figure that answers:
    'Within each passenger class, how do survival rates compare between women and men across age groups?'
    """
    tab = survival_demographics(df)

    # A faceted bar chart: x = age_group, y = survival_rate, color = Sex, facet per Pclass
    fig = px.bar(
        tab,
        x="age_group",
        y="survival_rate",
        color="Sex",
        barmode="group",
        facet_col="Pclass",
        facet_col_wrap=3,
        category_orders={"age_group": list(AGE_CATS.categories), "Pclass": [1, 2, 3]},
        labels={
            "age_group": "Age Group",
            "survival_rate": "Survival Rate",
            "Sex": "Sex",
            "Pclass": "Class"
        },
        title="Survival Rate by Age Group and Sex within Each Class"
    )
    fig.update_yaxes(tickformat=".0%", matches=None)
    fig.update_layout(
        legend_title_text="Sex",
        margin=dict(l=10, r=10, t=50, b=10),
        bargap=0.2,
        height=500
    )
    return fig



# Exercise 2

def family_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add family_size = SibSp + Parch + 1 and group by family_size and Pclass.
    Compute:
      - n_passengers
      - avg_fare
      - min_fare
      - max_fare
    Return a sorted, readable table.
    """
    df2 = _ensure_core_columns(df)
    df2 = _ordered_pclass(df2)  # makes Pclass a categorical 1<2<3

    df2["family_size"] = df2[["SibSp", "Parch"]].sum(axis=1, min_count=1) + 1

    # Drop rows lacking needed fields
    tmp = df2.dropna(subset=["family_size", "Pclass", "Fare"]).copy()

    grouped = (
        tmp
        .groupby(["Pclass", "family_size"], observed=True, as_index=False)
        .agg(
            n_passengers=("Fare", "size"),
            avg_fare=("Fare", "mean"),
            min_fare=("Fare", "min"),
            max_fare=("Fare", "max"),
        )
        .sort_values(["Pclass", "family_size"])
        .reset_index(drop=True)
    )
    return grouped



def last_names(df: pd.DataFrame) -> pd.Series:
    """
    Extract last names from the 'Name' column and return a value_counts Series:
      index = last name, value = count
    """
    if "Name" not in df.columns:
        return pd.Series(dtype=int)

    def extract_last(name: str) -> str:
        if not isinstance(name, str):
            return np.nan
        # Titanic names often look like "Surname, Title. Given Names"
        # Take text before the first comma, strip quotes/parentheses/whitespace.
        last = name.split(",", 1)[0]
        last = re.sub(r'["\'() ]+', " ", last).strip()
        return last

    last_series = df["Name"].map(extract_last)
    counts = last_series.value_counts(dropna=True)
    return counts


def visualize_families(df: pd.DataFrame):
    """
    Plotly figure that answers:
    'How does average fare change with family size across passenger classes,
     and where are the largest groups (by count)?'
    """
    tab = family_groups(df)
    # Scatter/line hybrid: show trend and group sizes
    fig = px.line(
        tab,
        x="family_size",
        y="avg_fare",
        color="Pclass",
        markers=True,
        category_orders={"Pclass": [1, 2, 3]},
        labels={
            "family_size": "Family Size",
            "avg_fare": "Average Fare",
            "Pclass": "Class"
        },
        title="Average Fare vs. Family Size by Class"
    )
    # Add marker size for n_passengers to convey group prevalence
    fig.update_traces(mode="lines+markers")
    # Workaround to add size: replot scatter on top
    scatter = px.scatter(
        tab,
        x="family_size",
        y="avg_fare",
        color="Pclass",
        size="n_passengers",
        category_orders={"Pclass": [1, 2, 3]},
    )
    for tr in scatter.data:
        fig.add_trace(tr)

    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=500)
    return fig



# Bonus

def determine_age_division(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add older_passenger: Age > median Age within each Pclass.
    Boolean True/False; False also for = median or if Age is missing.
    Return the updated DataFrame.
    """
    df3 = _ensure_core_columns(df)
    df3 = _ordered_pclass(df3)

    # Compute class-wise median age using transform so the series aligns per row
    class_median = df3.groupby("Pclass")["Age"].transform("median")
    df3["older_passenger"] = (df3["Age"] > class_median) & df3["Age"].notna() & class_median.notna()
    return df3


def visualize_age_division(df: pd.DataFrame):
    """
    Explore how this age division relates to outcomes:
    'Within each class, how does survival rate differ for older vs. not-older passengers, by sex?'
    """
    df4 = determine_age_division(df)
    # Build a tidy table of survival rates by Pclass x Sex x older_passenger
    tab = (
        df4
        .groupby(["Pclass", "Sex", "older_passenger"], as_index=False)
        .agg(
            n=("Survived", "size"),
            survived=("Survived", lambda s: np.nansum(s == 1))
        )
    )
    tab["survival_rate"] = tab["survived"] / tab["n"]
    tab = tab.sort_values(["Pclass", "Sex", "older_passenger"]).reset_index(drop=True)

    fig = px.bar(
        tab,
        x="Sex",
        y="survival_rate",
        color="older_passenger",
        barmode="group",
        facet_col="Pclass",
        facet_col_wrap=3,
        category_orders={"Pclass": [1, 2, 3]},
        labels={
            "Sex": "Sex",
            "survival_rate": "Survival Rate",
            "older_passenger": "Older than Class Median?",
            "Pclass": "Class"
        },
        title="Survival Rate by Sex and Age-Division (Older vs Not) within Each Class"
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=500)
    return fig
