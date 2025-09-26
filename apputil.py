import pandas as pd
import plotly.express as px

# =============================================================================
# Exercise 1: Survival Patterns
# =============================================================================

def survival_demographics(df):
    """
    Analyzes survival patterns by classifying passengers into age groups
    and calculating survival rates for each combination of class, sex,
    and age group.

    Returns:
        pd.DataFrame: A table with survival statistics.
    """
    # Define age bins and labels for categorization
    bins = [0, 12, 19, 59, df['Age'].max()]
    labels = ['Child (0-12)', 'Teen (13-19)', 'Adult (20-59)', 'Senior (60+)']
    
    # Create the 'age_group' column
    df['age_group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)

    # Group by class, sex, and age group to calculate statistics
    demographics_df = df.groupby(['Pclass', 'Sex', 'age_group'], observed=True).agg(
        n_passengers=('PassengerId', 'count'),
        n_survivors=('Survived', 'sum')
    ).reset_index()

    # Calculate the survival rate for each group
    demographics_df['survival_rate'] = demographics_df['n_survivors'] / demographics_df['n_passengers']

    # Sort the results for better readability
    demographics_df = demographics_df.sort_values(by=['Pclass', 'Sex', 'age_group'])

    return demographics_df

def visualize_demographic(df):
    """
    Creates a Plotly bar chart to visualize survival rates by class, sex,
    and age group.

    Returns:
        plotly.graph_objects.Figure: The generated Plotly figure.
    """
    # Get the processed data from the analysis function
    survival_data = survival_demographics(df)

    # Create the figure
    fig = px.bar(
        survival_data,
        x='Pclass',
        y='survival_rate',
        color='age_group',
        facet_col='Sex',
        title='Survival Rate by Class, Sex, and Age Group',
        labels={
            'Pclass': 'Passenger Class',
            'survival_rate': 'Survival Rate',
            'age_group': 'Age Group'
        },
        category_orders={"Pclass": [1, 2, 3], "Sex": ["female", "male"]}
    )
    # Standardize the y-axis for rates (0 to 1)
    fig.update_yaxes(range=[0, 1])
    return fig

# =============================================================================
# NEW: Alternative Visualization Functions
# =============================================================================

### 1. Sunburst Chart
def visualize_sunburst(df):
    """
    Creates a sunburst chart to visualize the hierarchy of passenger demographics
    and their corresponding survival rates.
    """
    # Get the processed demographic data
    survival_data = survival_demographics(df)
    
    # Ensure all data in the path is a string for Plotly
    survival_data['Pclass'] = 'Class ' + survival_data['Pclass'].astype(str)

    # Create the sunburst chart
    fig = px.sunburst(
        survival_data,
        path=['Pclass', 'Sex', 'age_group'], # Defines the hierarchy of rings
        values='n_passengers',                # Size of segments based on number of passengers
        color='survival_rate',                # Color of segments based on survival rate
        color_continuous_scale='viridis',     # A pleasant color scale
        range_color=[0,1],                    # Standardize color scale from 0 to 1
        title='Titanic Survival by Demographic Hierarchy',
        hover_data={'survival_rate': ':.2f'}  # Format hover data
    )
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    return fig

### 2. Treemap
def visualize_treemap(df):
    """
    Creates a treemap to visualize passenger demographics, where rectangle size
    indicates group size and color indicates survival rate.
    """
    # Get the processed demographic data
    survival_data = survival_demographics(df)

    # Ensure all data in the path is a string for Plotly
    survival_data['Pclass'] = 'Class ' + survival_data['Pclass'].astype(str)

    # Create the treemap
    fig = px.treemap(
        survival_data,
        path=[px.Constant("All Passengers"), 'Pclass', 'Sex', 'age_group'], # Defines the hierarchy
        values='n_passengers',                  # Size of rectangles based on passenger count
        color='survival_rate',                  # Color of rectangles based on survival rate
        color_continuous_scale='Reds',          # Use a red color scale for intensity
        range_color=[0,1],                      # Standardize color scale
        title='Titanic Survival by Demographic Group Size',
        hover_data={'survival_rate': ':.2f'}    # Format hover data
    )
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    return fig

### 3. Heatmap
def visualize_heatmap(df):
    """
    Creates a heatmap to show survival rates across passenger class and
    other demographic groups.
    """
    # Get the processed demographic data
    survival_data = survival_demographics(df)
    
    # Combine Sex and Age Group for one of the axes
    survival_data['Demographic'] = survival_data['Sex'] + ' (' + survival_data['age_group'].astype(str) + ')'
    
    # Pivot the data to create a matrix suitable for a heatmap
    heatmap_data = survival_data.pivot_table(
        index='Pclass', 
        columns='Demographic', 
        values='survival_rate'
    )

    # Create the heatmap
    fig = px.imshow(
        heatmap_data,
        text_auto=".2f", # Display the survival rate on each cell, formatted to 2 decimal places
        aspect="auto",
        color_continuous_scale='RdYlGn', # Red-Yellow-Green scale is intuitive for rates
        range_color=[0,1],
        title='Heatmap of Survival Rates by Class and Demographic'
    )
    fig.update_xaxes(title_text='Demographic Group')
    fig.update_yaxes(title_text='Passenger Class')
    return fig
# =============================================================================
# Exercise 2: Family Size and Wealth
# =============================================================================

def family_groups(df):
    """
    Analyzes the relationship between family size, passenger class, and ticket fare.

    Returns:
        pd.DataFrame: A table with statistics on family size and fares.
    """
    # Create 'family_size' column (SibSp + Parch + self)
    df['family_size'] = df['SibSp'] + df['Parch'] + 1

    # Group by class and family size to calculate fare statistics
    family_df = df.groupby(['Pclass', 'family_size']).agg(
        n_passengers=('PassengerId', 'count'),
        avg_fare=('Fare', 'mean'),
        min_fare=('Fare', 'min'),
        max_fare=('Fare', 'max')
    ).reset_index()

    # Sort the results for better readability
    family_df = family_df.sort_values(by=['Pclass', 'family_size'])

    return family_df

def last_names(df):
    """
    Extracts the last name of each passenger and returns the count for each name.

    Returns:
        pd.Series: A series with last names as the index and their counts as values.
    """
    last_names_series = df['Name'].apply(lambda name: name.split(',')[0])
    return last_names_series.value_counts()

def visualize_families(df):
    """
    Creates a Plotly scatter plot to visualize the relationship between family
    size, wealth (fare), and passenger class.

    Returns:
        plotly.graph_objects.Figure: The generated Plotly figure.
    """
    # Get the processed data
    family_data = family_groups(df)

    # Create the scatter plot, with bubble size representing passenger count
    fig = px.scatter(
        family_data,
        x='family_size',
        y='avg_fare',
        color='Pclass',
        size='n_passengers',
        hover_name='Pclass',
        hover_data=['min_fare', 'max_fare'],
        title='Average Fare vs. Family Size by Passenger Class',
        labels={
            'family_size': 'Family Size (including self)',
            'avg_fare': 'Average Ticket Fare ($)',
            'n_passengers': 'Number of Passengers',
            'Pclass': 'Passenger Class'
        },
        category_orders={"Pclass": [1, 2, 3]}
    )
    return fig

def visualize_families_line(df):
    """
    Creates a faceted line plot to show the trend of average fare by
    family size for each passenger class.
    """
    # Get the processed data
    family_data = family_groups(df)

    # Create the faceted line plot
    fig = px.line(
        family_data,
        x='family_size',
        y='avg_fare',
        facet_col='Pclass',  # Creates a separate plot for each class
        markers=True,        # Adds dots on each data point
        title='Average Fare Trend by Family Size for Each Passenger Class',
        labels={
            'family_size': 'Family Size',
            'avg_fare': 'Average Ticket Fare ($)',
            'Pclass': 'Passenger Class'
        }
    )
    # This ensures each subplot can have its own y-axis range, which is better for trends
    fig.update_yaxes(matches=None)
    return fig

# =============================================================================
# Bonus Question
# =============================================================================

def determine_age_division(df):
    """
    Adds a boolean column 'older_passenger' that indicates if a passenger's
    age is above the median age for their specific passenger class.

    Returns:
        pd.DataFrame: The DataFrame with the new 'older_passenger' column.
    """
    # Use transform to get the median age for each passenger's class
    median_ages = df.groupby('Pclass')['Age'].transform('median')
    
    # Create the boolean column
    df['older_passenger'] = df['Age'] > median_ages
    return df

def visualize_family_size(df): # Note: Named to match the original app.py
    """
    Visualizes survival rates based on whether a passenger's age is above
    or below the median for their class.

    Returns:
        plotly.graph_objects.Figure: The generated Plotly figure.
    """
    # Get the processed data using a copy to avoid modifying the original df
    age_data = determine_age_division(df.copy())

    # Calculate survival rates for the new groups
    bonus_df = age_data.groupby(['Pclass', 'older_passenger'])['Survived'].mean().reset_index()
    bonus_df['older_passenger'] = bonus_df['older_passenger'].map({
        True: 'Above Median Age',
        False: 'Below/At Median Age'
    })

    # Create a grouped bar chart for comparison
    fig = px.bar(
        bonus_df,
        x='Pclass',
        y='Survived',
        color='older_passenger',
        barmode='group',
        title='Survival Rate by Class and Age Relative to Class Median',
        labels={
            'Pclass': 'Passenger Class',
            'Survived': 'Survival Rate',
            'older_passenger': 'Age vs. Class Median'
        },
        category_orders={"Pclass": [1, 2, 3]}
    )
    # Standardize the y-axis for rates (0 to 1)
    fig.update_yaxes(range=[0, 1])
    return fig