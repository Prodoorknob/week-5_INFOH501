import streamlit as st
import pandas as pd
from apputil import *

# It's a good practice to cache data loading to improve performance
@st.cache_data
def load_data():
    """Loads the Titanic dataset from a URL."""
    df = pd.read_csv('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv')
    return df

# Load Titanic dataset
df = load_data()


# --- Exercise 1 ---
st.write(
    '''
    ## Titanic Visualization 1: Survival Patterns
    '''
)

st.write(
    "**Question:** How did survival rates differ across passenger class, sex, and age group? "
    "Specifically, were women and children in first class significantly more likely to "
    "survive than men in third class?"
)

# Generate and display the figure for Exercise 1
# fig1 = visualize_demographic(df)
# st.plotly_chart(fig1, use_container_width=True)

# # Import the new functions from apputil
# #rom apputil import visualize_sunburst, visualize_treemap, visualize_heatmap

# st.write(
#     '''
#     # Alternative Visualization 1: Sunburst Chart
#     '''
# )
# fig_sunburst = visualize_sunburst(df)
# st.plotly_chart(fig_sunburst, use_container_width=True)


# st.write(
#     '''
#     # Alternative Visualization 2: Treemap
#     '''
# )
# fig_treemap = visualize_treemap(df)
# st.plotly_chart(fig_treemap, use_container_width=True)


# st.write(
#     '''
#     # Alternative Visualization 3: Heatmap
#     '''
# )
fig_heatmap = visualize_heatmap(df)
st.plotly_chart(fig_heatmap, use_container_width=True)


# --- Exercise 2 ---
st.write(
    '''
    ## Titanic Visualization 2: Family Size & Wealth
    '''
)

st.write(
    "**Question:** Is there a relationship between family size and wealth (as indicated by ticket fare) "
    "across different passenger classes?"
)

# Generate and display the figure for Exercise 2
fig2 = visualize_families_line(df)
st.plotly_chart(fig2, use_container_width=True)

st.write(
    "**Last Name Analysis:** The `family_size` column is calculated using the 'SibSp' and 'Parch' columns. "
    "An alternative approach is to count passengers with the same last name. However, these methods can differ. "
    "The `family_size` method is more precise for nuclear families, while counting last names might group "
    "extended family or unrelated individuals, and could miss family members with different last names."
)

# Optional: Display the top last names as an example
st.write("Top 10 most common last names on board:")
st.dataframe(last_names(df).head(10))


# --- Bonus Exercise ---
st.write(
    '''
    ## Titanic Visualization Bonus: Age & Survival
    '''
)

st.write(
    "**Analysis:** This chart explores whether being older or younger than the median age *for one's own passenger class* "
    "had an impact on survival rates."
)

# Generate and display the figure for the Bonus Exercise
fig3 = visualize_family_size(df)
st.plotly_chart(fig3, use_container_width=True)