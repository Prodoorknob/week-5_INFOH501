import plotly.express as px
import pandas as pd

# update/add code below ...
# apputil.py
import pathlib
import pandas as pd
import plotly.express as px

# -------- Data loading helpers --------
def _load_titanic() -> pd.DataFrame:
    """
    Load Titanic training data.
    Priority:
      1) ./train.csv (downloaded from Kaggle competition)
      2) Fallback to a public mirror if local file not found
    """
    local = pathlib.Path("train.csv")
    if local.exists():
        df = pd.read_csv(local)
    else:
        # Fallback mirror to keep the app working out-of-the-box
        df = pd.read_csv(
            "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        )

    # Standardize expected columns
    # Some mirrors use slightly different casing; align to Kaggle's schema.
    rename_map = {
        "survived": "Survived",
        "pclass": "Pclass",
        "sex": "Sex",
        "age": "Age",
        "sibsp": "SibSp",
        "parch": "Parch",
        "fare": "Fare",
        "embarked": "Embarked",
        "name": "Name",
        "ticket": "Ticket",
        "cabin": "Cabin",
        "passengerid": "PassengerId",
    }
    df = df.rename(columns=rename_map)

    # Create engineered columns often used in exercises
    df["FamilySize"] = df[["SibSp", "Parch"]].sum(axis=1) + 1  # include self
    df["Alone"] = (df["FamilySize"] == 1).map({True: "Alone", False: "With family"})

    # Clean basic types
    if "Pclass" in df.columns:
        df["Pclass"] = df["Pclass"].astype("Int64")
    if "Survived" in df.columns:
        df["Survived"] = df["Survived"].astype("Int64")

    return df


# Cache in module scope so multiple figures reuse the same frame
_DF = None
def _df() -> pd.DataFrame:
    global _DF
    if _DF is None:
        _DF = _load_titanic()
    return _DF


# -------- Visualizations referenced by app.py --------
def visualize_demographic():
    """
    Visualization 1:
    Survival rate by Sex and Pclass with Age distribution context.
    Returns a Plotly figure suitable for st.plotly_chart(...)
    """
    df = _df().copy()

    # Bin ages to make the faceting/bar more legible when many nulls exist
    age_bins = [0, 12, 18, 30, 45, 60, 80]
    df["AgeBand"] = pd.cut(df["Age"], bins=age_bins, include_lowest=True)

    # Use bar with percentage (normalized count) of Survived within groups
    # If Survived is missing in a mirror, fallback gracefully
    if "Survived" in df.columns:
        fig = px.histogram(
            df,
            x="Sex",
            color="Survived",
            barmode="group",
            facet_col="Pclass",
            facet_row="AgeBand",
            category_orders={"Sex": ["male", "female"], "Survived": [0, 1]},
            text_auto=True,
            barnorm="percent",
        )
        fig.update_layout(
            title="Survival % by Sex, Pclass, and Age Band",
            legend_title_text="Survived",
        )
    else:
        # Fallback: show population distribution if Survived not present
        fig = px.histogram(
            df,
            x="Sex",
            color="Pclass",
            barmode="group",
            facet_row="AgeBand",
            text_auto=True,
        )
        fig.update_layout(title="Passenger Distribution by Sex, Pclass, and Age Band")

    fig.for_each_annotation(lambda a: a.update(text=a.text.replace("Pclass=", "Class ")))
    fig.update_yaxes(title="Percent of group")
    fig.update_xaxes(title="")
    return fig


def visualize_families():
    """
    Visualization 2:
    Family structure vs survival:
      - x = FamilySize (clipped to keep chart readable)
      - color = Alone vs With family
      - facet by Embarked (if available)
    """
    df = _df().copy()
    df["FamilySizeClipped"] = df["FamilySize"].clip(upper=7)

    if "Survived" in df.columns:
        # Compute survival rate by family size & alone/with family
        grp = (
            df.dropna(subset=["FamilySizeClipped"])
              .groupby(["FamilySizeClipped", "Alone"], as_index=False)["Survived"]
              .mean()
              .rename(columns={"Survived": "SurvivalRate"})
        )
        fig = px.line(
            grp,
            x="FamilySizeClipped",
            y="SurvivalRate",
            color="Alone",
            markers=True,
        )
        fig.update_layout(
            title="Survival Rate vs Family Size",
            yaxis_tickformat=".0%",
        )
        fig.update_xaxes(title="Family Size (clipped at 7)")
        fig.update_yaxes(title="Survival Rate")
    else:
        # Fallback: show distribution by family size
        fig = px.histogram(
            df,
            x="FamilySizeClipped",
            color="Alone",
            barmode="group",
            text_auto=True,
        )
        fig.update_layout(title="Passenger Count by Family Size and Alone/With Family")
        fig.update_xaxes(title="Family Size (clipped at 7)")
        fig.update_yaxes(title="Passenger Count")

    return fig


def visualize_family_size():
    """
    Bonus Visualization:
    Survival by discrete buckets of FamilySize with Fare as a secondary cue.
    """
    df = _df().copy()
    df["FamilyBucket"] = pd.cut(
        df["FamilySize"],
        bins=[0, 1, 2, 4, 7, 11],
        labels=["1", "2", "3-4", "5-7", "8-11"],
        include_lowest=True,
    )

    if "Survived" in df.columns:
        grp = (
            df.dropna(subset=["FamilyBucket"])
              .groupby(["FamilyBucket"], as_index=False)["Survived"].mean()
              .rename(columns={"Survived": "SurvivalRate"})
        )
        fig = px.bar(
            grp,
            x="FamilyBucket",
            y="SurvivalRate",
            text="SurvivalRate",
        )
        fig.update_traces(texttemplate="%{text:.1%}", textposition="outside", cliponaxis=False)
        fig.update_layout(
            title="Survival Rate by Family Size Bucket",
            uniformtext_minsize=10,
            uniformtext_mode="hide",
        )
        fig.update_xaxes(title="Family Size Bucket")
        fig.update_yaxes(title="Survival Rate", tickformat=".0%", rangemode="tozero")
    else:
        fig = px.bar(
            df.groupby("FamilyBucket").size().reset_index(name="Count"),
            x="FamilyBucket",
            y="Count",
            text="Count",
            title="Passenger Count by Family Size Bucket",
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_xaxes(title="Family Size Bucket")
        fig.update_yaxes(title="Count", rangemode="tozero")

    return fig
