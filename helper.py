import numpy as np
import pandas as pd
import pycountry



def fetch_medal_tally(df, year, country):
    medal_df = df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'])
    flag = 0
    if year == 'Overall' and country == 'Overall':
        temp_df = medal_df
    if year == 'Overall' and country != 'Overall':
        flag = 1
        temp_df = medal_df[medal_df['region'] == country]
    if year != 'Overall' and country == 'Overall':
        temp_df = medal_df[medal_df['Year'] == int(year)]
    if year != 'Overall' and country != 'Overall':
        temp_df = medal_df[(medal_df['Year'] == year) & (medal_df['region'] == country)]

    if flag == 1:
        x = temp_df.groupby('Year').sum()[['Gold', 'Silver', 'Bronze']].sort_values('Year').reset_index()
    else:
        x = temp_df.groupby('region').sum()[['Gold', 'Silver', 'Bronze']].sort_values('Gold',
                                                                                      ascending=False).reset_index()

    x['total'] = x['Gold'] + x['Silver'] + x['Bronze']

    x['Gold'] = x['Gold'].astype('int')
    x['Silver'] = x['Silver'].astype('int')
    x['Bronze'] = x['Bronze'].astype('int')
    x['total'] = x['total'].astype('int')

    return x


def country_year_list(df):
    years = df['Year'].unique().tolist()
    years.sort()
    years.insert(0, 'Overall')

    country = np.unique(df['region'].dropna().values).tolist()
    country.sort()
    country.insert(0, 'Overall')

    return years,country

def data_over_time(df, col):
    # Remove duplicate entries for the combination of Year and the given column (e.g., region)
    temp_df = df.drop_duplicates(['Year', col])

    # Group by Year and count the number of unique entries per year
    data = temp_df.groupby('Year')[col].nunique().reset_index()

    # Rename the columns for clarity
    data.rename(columns={col: col, 'Year': 'Edition'}, inplace=True)

    return data



def most_successful(df, sport):
    # Remove rows where Medal is NaN
    temp_df = df.dropna(subset=['Medal'])

    # If a specific sport is selected, filter by it
    if sport != 'Overall':
        temp_df = temp_df[temp_df['Sport'] == sport]

    # Get top 15 athletes based on medal count
    top_athletes = temp_df['Name'].value_counts().reset_index().head(15)
    top_athletes.columns = ['Name', 'Medals']  # Rename columns properly

    # Merge with original dataframe to get Sport and Region info
    merged = top_athletes.merge(
        df[['Name', 'Sport', 'region']].drop_duplicates(),
        on='Name',
        how='left'
    )

    return merged



def yearwise_medal_tally(df,country):
    temp_df = df.dropna(subset=['Medal'])
    temp_df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'], inplace=True)

    new_df = temp_df[temp_df['region'] == country]
    final_df = new_df.groupby('Year').count()['Medal'].reset_index()

    return final_df

def country_event_heatmap(df,country):
    temp_df = df.dropna(subset=['Medal'])
    temp_df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'], inplace=True)

    new_df = temp_df[temp_df['region'] == country]

    pt = new_df.pivot_table(index='Sport', columns='Year', values='Medal', aggfunc='count').fillna(0)
    return pt


def most_successful_countrywise(df, country):
    temp_df = df.dropna(subset=['Medal'])
    temp_df = temp_df[temp_df['region'] == country]

    # Get top 10 medal-winning athletes
    top_athletes = temp_df['Name'].value_counts().reset_index().head(10)
    top_athletes.columns = ['Name', 'Medals']  # Rename columns properly

    # Merge with original DataFrame to get Sport
    top_athletes = top_athletes.merge(
        df[['Name', 'Sport']].drop_duplicates(), on='Name', how='left'
    )

    return top_athletes

def get_country_code(country_name):
    try:
        country = pycountry.countries.search_fuzzy(country_name)[0]
        return country.alpha_2.lower()
    except:
        return None

def weight_v_height(df,sport):
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])
    athlete_df['Medal'].fillna('No Medal', inplace=True)
    if sport != 'Overall':
        temp_df = athlete_df[athlete_df['Sport'] == sport]
        return temp_df
    else:
        return athlete_df

def men_vs_women(df):
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])

    men = athlete_df[athlete_df['Sex'] == 'M'].groupby('Year').count()['Name'].reset_index()
    women = athlete_df[athlete_df['Sex'] == 'F'].groupby('Year').count()['Name'].reset_index()

    final = men.merge(women, on='Year', how='left')
    final.rename(columns={'Name_x': 'Male', 'Name_y': 'Female'}, inplace=True)

    final.fillna(0, inplace=True)

    return final





def compare_two_countries(df, country1, country2):
    # Ensure the 'Medal' column has no NaN values
    df = df.dropna(subset=['Medal'])
    # Remove duplicates based on relevant columns
    df = df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'])

    # Get medal count per year for both countries
    df1 = df[df['region'] == country1].groupby('Year')['Medal'].count().reset_index().rename(
        columns={'Medal': 'Country1_Medals'})
    df2 = df[df['region'] == country2].groupby('Year')['Medal'].count().reset_index().rename(
        columns={'Medal': 'Country2_Medals'})

    # Merge the two dataframes based on Year and fill missing values with 0
    merged_df = pd.merge(df1, df2, on='Year', how='outer').fillna(0).sort_values('Year')

    # Convert the medal counts to integers
    merged_df['Country1_Medals'] = merged_df['Country1_Medals'].astype(int)
    merged_df['Country2_Medals'] = merged_df['Country2_Medals'].astype(int)

    return merged_df



