
# python -m streamlit run app.py

import streamlit as st
import pandas as pd
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
from sklearn.linear_model import LinearRegression
import wikipedia
import random
import pycountry
import plotly.express as px




# Load only the base dataset
df = pd.read_csv('athlete_events_compressed.zip')

# Load region data
region_df = pd.read_csv('noc_regions.csv')

# Preprocess
df = preprocessor.preprocess(df, region_df)




# ✅ Remove unwanted sports
unwanted_sports = ['Art Competitions','Triathlon', 'Alpinism', 'Aeronautics', 'Sailing','Judo','Synchronized Swimming', 'Modern Pentathlon','Jeu De Paume','Motorboating','Polo','Racquets','Roque']
df = df[~df['Sport'].isin(unwanted_sports)]

# Video links per sport
sport_video_dict = {
    "Basketball": "https://www.youtube.com/watch?v=oyjYgmsM00Q",
    "Football": "https://www.youtube.com/watch?v=nT5qyrxoqsA",
    "Tennis": "https://www.youtube.com/watch?v=S9DnaBlhlVI",
    "Swimming": "https://www.youtube.com/watch?v=4jSHwgpFJe8",
    "Gymnastics": "https://www.youtube.com/watch?v=o2yHwuB7F00",
    "Archery": "https://www.youtube.com/watch?v=5U53PllOWvU",
    "Badminton": "https://www.youtube.com/watch?v=tAS7rOKtpgQ",
    "Baseball": "https://www.youtube.com/watch?v=57mc7Df7Arw",
    "Boxing": "https://www.youtube.com/watch?v=7EMa8hMHcXI",
    "Athletics": "https://www.youtube.com/watch?v=97iKteyVj1A",
    "Basque Pelota": "https://www.youtube.com/watch?v=B6Zcv7wsbbU",
    "Canoeing": "https://www.youtube.com/watch?v=DKNqpr3NUb8",
    "Beach Volleyball": "https://www.youtube.com/watch?v=3ON-ZyA0G9k",
    "Cricket": "https://www.youtube.com/watch?v=yXIJcKpFlV4",
    "Croquet": "https://www.youtube.com/watch?v=xXGm639Z7Z8",
    "Cycling":"https://www.youtube.com/watch?v=qUeyxDVtlWY",
    "Diving":"https://www.youtube.com/watch?v=OopNADRstu4",
    "Equestrianism":"https://www.youtube.com/watch?v=UKFvZSBViUE",
    "Fencing":"https://www.youtube.com/watch?v=Q6-aH-op4g4",
    "Figure Skating": "https://www.youtube.com/watch?v=ultolnvZpqw",
    "Golf": "https://www.youtube.com/watch?v=99nN7WWNF1Q",
    "Handball":"https://www.youtube.com/watch?v=PcBwK9NTqNw",
    "Hockey":"https://www.youtube.com/watch?v=6CjVQ1AtudQ&t=33s",
    "Ice Hockey":"https://www.youtube.com/watch?v=H_70vAiyyXM",
    "Volleyball":"https://www.youtube.com/watch?v=907TGg-CXYc",
    "Lacrosse":"https://www.youtube.com/watch?v=O03TuYCQ3JY",
    "Rhythmic Gymnastics":"https://www.youtube.com/watch?v=fFgyLS5fbW0",
    "Rowing":"https://www.youtube.com/watch?v=rz3UmSc8x_E",
    "Rugby":"https://www.youtube.com/watch?v=keHYLxeQaLU",
    "Rugby Sevens":"https://www.youtube.com/watch?v=1e894rFZvqQ",
    "Shooting":"https://www.youtube.com/watch?v=zyrG-iXDVC8",
    "Softball":"https://www.youtube.com/watch?v=YLU6W6AYQto",
    "Table Tennis":"https://www.youtube.com/watch?v=lwOwIBWkxl4&t=37s",
    "Taekwondo":"https://www.youtube.com/watch?v=Fw0_mQI1lkc",
    "Trampolining":"https://www.youtube.com/watch?v=VqWFNvonmN4",
    "Tug-Of-War":"https://www.youtube.com/watch?v=WOFvWk35sag",
    "Water Polo":"https://www.youtube.com/watch?v=g63LpPuDaxE",
    "Weightlifting":"https://www.youtube.com/watch?v=l8oxCtwQdm0",
    "Wrestling":"https://www.youtube.com/watch?v=iGWmUCW82P0",

    # Add other sports and their corresponding video links here
}



# Sidebar
st.sidebar.title("Olympics Analysis")
st.sidebar.image("https://e7.pngegg.com/pngimages/1020/402/png-clipart-2024-summer-olympics-brand-circle-area-olympic-rings-olympics-logo-text-sport.png")
user_menu = st.sidebar.radio(
    'Select an Option',
    (
        'Olympic Country Summary', 'Medal Tally', "Top Athletes Explorer",'Overall Analysis', 'Country-wise Analysis',
        'Athlete wise Analysis', "Animated Medal Chart",'Sport Video', 'Medal Predictor',
        "Top Athletes Explorer",'Historical World Rankings', 'Country Comparison','Dominant Countries by Sport','Olympic Moments','Download Country History',
    )
)


# --- Initialize session state ---
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'attempts' not in st.session_state:
    st.session_state.attempts = 0
if 'chosen_country' not in st.session_state:
    st.session_state.chosen_country = None
if 'show_result' not in st.session_state:
    st.session_state.show_result = False


if user_menu == "Top Athletes Explorer":
    st.title("🌍 Top Athletes Explorer")
    st.markdown("Discover Olympic athletes from any era with their images and short biographies. Powered by Wikipedia.")

    # Athlete input
    athlete_name = st.text_input("🔎 Enter Athlete Name", "")

    # Drop-in: robust Wikipedia lookup (English), disambiguation handling, and page-specific images
    import wikipedia
    from wikipedia.exceptions import DisambiguationError, PageError
    from io import BytesIO
    from PIL import Image
    import requests

    wikipedia.set_lang("en")

    def search_candidates(query, max_results=5):
        try:
            return wikipedia.search(query, results=max_results)
        except Exception:
            return []

    def resolve_title(query):
        candidates = search_candidates(query, max_results=5)
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        # Let user pick if multiple
        return st.selectbox("Multiple pages found. Choose the athlete:", candidates)

    def load_page(title):
        try:
            return wikipedia.page(title)
        except (DisambiguationError, PageError):
            return None
        except Exception:
            return None

    def get_image_candidates(page):
        """Return a list of candidate image URLs from the page, excluding logos/icons."""
        if not page:
            return []
        cands = []
        for img in page.images:
            low = img.lower()
            if any(low.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".gif")) and \
               "logo" not in low and "icon" not in low:
                cands.append(img)
        return cands

    def safe_load_image(url, timeout=10):
        """Try loading an image with a browser-like user agent. Returns a PIL Image or None."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115 Safari/537.36"
            }
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content))
        except Exception:
            return None

    def get_summary(page, title):
        if page and getattr(page, "summary", None):
            return page.summary
        try:
            # Fallback to a short wiki summary if page.summary isn't available
            return wikipedia.summary(title, sentences=3)
        except Exception:
            # Final fallback: use first paragraph from page content if available
            try:
                if page and getattr(page, "content", None):
                    content = page.content
                    first_para = content.split("\n")[0]
                    return first_para
            except Exception:
                pass
        return None

    # Placeholder image when nothing loads
    PLACEHOLDER = "https://upload.wikimedia.org/wikipedia/commons/6/69/No_image_available.png"

    if athlete_name:
        chosen_title = resolve_title(athlete_name)
        if not chosen_title:
            st.info("✍️ No Wikipedia page found for that name. Try a more specific term (e.g., 'Usain Bolt').")
        else:
            page = load_page(chosen_title)
            st.markdown(f"## 🏅 {chosen_title}")

            # Try multiple candidate images until one loads
            image_loaded = False
            if page:
                for img_url in get_image_candidates(page):
                    img = safe_load_image(img_url)
                    if img:
                        st.image(img, caption=chosen_title, width=300)
                        image_loaded = True
                        break

            if not image_loaded:
                # Try placeholder image
                pl = safe_load_image(PLACEHOLDER)
                if pl:
                    st.image(pl, caption=f"{chosen_title} (image not loadable)", width=300)
                else:
                    st.info("ℹ️ No loadable image found for this page.")

            # Summary with fallback
            summary = get_summary(page, chosen_title)
            if summary:
                st.write(summary)
            else:
                st.warning("⚠️ No biography found on Wikipedia.")
    else:
        st.info("✍️ Type an athlete name above to explore their biography & image.")


# Historical World Rankings Animation
if user_menu == 'Historical World Rankings':
    st.title("🌍 Historical World Rankings Animation")
    st.markdown("Visualize how Olympic medals have been distributed across the world over time.")

    # Select medal type
    medal_filter = st.radio("Choose Medal Type", ['Total', 'Gold', 'Silver', 'Bronze'], horizontal=True)

    # Filter base medal dataframe
    medal_df = df[df['Medal'].notna()]

    # Apply specific medal filter
    if medal_filter != 'Total':
        medal_df = medal_df[medal_df['Medal'] == medal_filter]

    # Group by Year and Country
    animation_df = (
        medal_df
        .groupby(['Year', 'region'])['Medal']
        .count()
        .reset_index()
        .rename(columns={'region': 'Country', 'Medal': 'Medal Count'})
    )

    # Build animated choropleth
    fig = px.choropleth(
        animation_df,
        locations="Country",
        locationmode="country names",
        color="Medal Count",
        hover_name="Country",
        animation_frame="Year",
        color_continuous_scale="Oranges" if medal_filter == "Bronze" else
                              "Greys" if medal_filter == "Silver" else
                              "YlOrBr" if medal_filter == "Gold" else
                              "Plasma",
        title=f"Olympic {medal_filter} Medal Distribution Over Time" if medal_filter != "Total" else "Total Olympic Medals Over Time"
    )

    fig.update_layout(
        geo=dict(showframe=False, showcoastlines=False),
        height=600,
        margin={"r":0,"t":50,"l":0,"b":0}
    )

    st.plotly_chart(fig)

# ⏱️ Title and Sidebar View Selector
if user_menu == 'Olympic eventsss':
    st.title("🗓️ This Day in Olympic History")
    st.sidebar.header("📆 Olympic Daily View")
    view = st.sidebar.selectbox("Select View", ["Historical Events", "Athlete Birthdays"])

    st.markdown(f"### 📅 {today_month} {today_day}")

    query = f"{today_month} {today_day}"

    # 👇 View: Historical Events
    if view == "Historical Events":
        try:
            page = wikipedia.page(query)
            content = page.content
            olympic_events = [line for line in content.split('\n') if 'Olympic' in line or 'Olympics' in line]
            if olympic_events:
                st.subheader("🏅 Notable Olympic Events:")
                for event in olympic_events[:10]:
                    st.markdown(f"- {event}")
            else:
                st.info("ℹ️ No Olympic events found for today.")
        except Exception as e:
            st.error("❌ Could not retrieve Olympic events.")
            st.text(str(e))

    # 👇 View: Athlete Birthdays
    elif view == "Athlete Birthdays":
        try:
            birthday_query = f"Births on {today_month} {today_day}"
            page = wikipedia.page(birthday_query)
            content = page.content
            olympians = [line for line in content.split('\n') if 'Olympic' in line or 'Olympics' in line]
            if olympians:
                st.subheader("🎂 Olympic Athlete Birthdays:")
                for person in olympians[:10]:
                    st.markdown(f"- {person}")
            else:
                st.info("ℹ️ No Olympic athlete birthdays found today.")
        except Exception as e:
            st.error("❌ Could not retrieve athlete birthdays.")
            st.text(str(e))


# Function to get country code for flag
def get_country_code(country_name):
    try:
        match = pycountry.countries.search_fuzzy(country_name)
        return match[0].alpha_2.lower()
    except:
        return None

# Menu option
if user_menu == 'Olympic Country Summary':
    st.title("📘 Country-Wise Olympic Summary")

    # 🧭 Sidebar country selector
    st.sidebar.markdown("### 🌍 Explore by Country")
    country_list = sorted(df['region'].dropna().unique())
    selected_country = st.sidebar.selectbox("Select a Country", country_list)

    st.subheader(f"🏅 {selected_country} at the Olympics")

    # 🌍 Flag
    country_code = get_country_code(selected_country)
    if country_code:
        flag_url = f"https://flagcdn.com/w320/{country_code}.png"
        st.image(flag_url, width=300, caption=f"Flag of {selected_country}")
    else:
        st.info("⚠️ Flag not found for this country.")

    # 📘 Wikipedia summary
    try:
        summary = wikipedia.summary(f"{selected_country} at the Olympics", sentences=5, auto_suggest=False)
        st.info(summary)
    except wikipedia.DisambiguationError as e:
        st.warning("🔎 Multiple Wikipedia pages found. Try refining your search.")
        st.text(f"Suggestions: {', '.join(e.options[:5])}")
    except wikipedia.PageError:
        st.error("❌ No page found for this country’s Olympic history.")
    except Exception as e:
        st.error("⚠️ An unexpected error occurred.")
        st.text(str(e))

    #  Top 5 famous athletes with images and bios
    st.markdown("---")
    st.subheader(f" Famous Athletes from {selected_country}")

    medal_df = df[(df['region'] == selected_country) & (df['Medal'].notna())]
    top_athletes = medal_df['Name'].value_counts().head(15).index.tolist()

    shown = 0
    for athlete in top_athletes:
        if shown >= 5:
            break
        try:
            page = wikipedia.page(athlete, auto_suggest=False)
            summary = wikipedia.summary(athlete, sentences=2)
            image_url = next(
                (img for img in page.images if img.lower().endswith(('.jpg', '.png')) and "logo" not in img.lower()),
                None
            )
            if image_url:
                st.image(image_url, width=200)
            st.markdown(f"**{athlete}**")
            st.markdown(summary)
            shown += 1
        except:
            continue  # Skip broken/ambiguous/missing athlete entries

    if shown == 0:
        st.info(f"No valid athlete biographies available for {selected_country}.")


if user_menu == 'Download Country History':
    st.title("📥 Download Olympic History for a Country")

    selected_country = st.selectbox("Choose Country", sorted(df['region'].dropna().unique()))
    history_df = df[df['region'] == selected_country].drop_duplicates()

    st.write(f"Olympic data for {selected_country}:")
    st.dataframe(history_df[['Year', 'City', 'Sport', 'Event', 'Medal']])

    st.download_button("📁 Download CSV", history_df.to_csv(index=False), f"{selected_country}_olympic_history.csv")



if user_menu == 'Olympic Moments':
    st.title("🎥 Famous Olympic Moments")

    videos = {
        "Usain Bolt 100m 2008": "https://www.youtube.com/watch?v=2O7K-8G2nwU",
        "Michael Phelps Beijing 2016": "https://www.youtube.com/watch?v=UmIYanq5gH8",
        "Opening Ceremony Tokyo 2020": "https://www.youtube.com/watch?v=UJyReGFKQU8&list=RDUJyReGFKQU8&start_radio=1",
        "Men's 4x100m Final Paris Champions": "https://www.youtube.com/watch?v=OFk8-4S5sD4",
        "A New World Record! | Men's 100m Freestyle":"https://www.youtube.com/watch?v=q14W1uCJag4",
        "Cycling Men's Road Race":"https://www.youtube.com/watch?v=H6Om072dfbU",
        " WORLD RECORD! | Men's Pole":"https://www.youtube.com/watch?v=P0HLeKJBqJk",
        "🥇 Neeraj Chopra wins historic gold for India":"https://www.youtube.com/watch?v=rW_fwcmyIfk&t=22s",
    }

    for title, url in videos.items():
        st.subheader(title)
        st.video(url)


if user_menu == 'Dominant Countries by Sport':
    st.title("🥇 Dominant Countries per Sport")

    sports = sorted(df['Sport'].dropna().unique())
    selected = st.selectbox("Select a Sport", sports)

    top_countries = df[(df['Sport'] == selected) & (df['Medal'].notna())]
    top = top_countries['region'].value_counts().head(10).reset_index()
    top.columns = ['Country', 'Total Medals']

    fig = px.bar(top, x='Country', y='Total Medals', color='Total Medals',
                 title=f"Top Countries in {selected}")
    st.plotly_chart(fig)

if user_menu == 'Country Comparison':
    st.title("📊 Country Medal Comparison")

    countries = sorted(df['region'].dropna().unique())
    country1 = st.selectbox("Select Country 1", countries)
    country2 = st.selectbox("Select Country 2", countries, index=1)

    compare_df = helper.compare_two_countries(df, country1, country2)

    fig = px.line(compare_df, x='Year', y=['Country1_Medals', 'Country2_Medals'],
                  labels={'value': 'Medals', 'variable': 'Country'},
                  title=f"Medal Comparison: {country1} vs {country2}")
    st.plotly_chart(fig)

if user_menu == "Animated Medal Chart":
    st.title("📊 Animated Olympic Medal Tally")

    # 🎯 Sidebar Year Selection
    st.sidebar.markdown("### 🕹️ Control Panel")
    st.sidebar.markdown("Adjust Olympic year to see medal leaderboard changes:")

    available_years = sorted(df['Year'].dropna().unique())
    min_year, max_year = int(min(available_years)), int(max(available_years))
    year = st.sidebar.slider("📅 Select Olympic Year", min_year, max_year, step=4, value=max_year)

    # 🎯 Optional: Medal Type filter
    medal_type = st.sidebar.radio("Medal Type", ['All', 'Gold', 'Silver', 'Bronze'], horizontal=False)

    # 🎯 Apply filters
    filtered_df = df[(df['Year'] <= year) & (df['Medal'].notna())]

    if medal_type != 'All':
        filtered_df = filtered_df[filtered_df['Medal'] == medal_type]

    if filtered_df.empty:
        st.warning(f"No medal data available up to {year}.")
    else:
        medal_counts = (
            filtered_df
            .groupby('region')['Medal']
            .count()
            .reset_index()
            .sort_values(by='Medal', ascending=False)
            .head(15)
        )
        medal_counts.columns = ['Country', 'Total Medals']

        fig = px.bar(
            medal_counts,
            x='Country',
            y='Total Medals',
            color='Total Medals',
            title=f"🏅 Top 15 Countries by {medal_type if medal_type != 'All' else 'All'} Medals (Up to {year})",
            color_continuous_scale='Turbo'
        )
        fig.update_layout(xaxis_title="Country", yaxis_title="Total Medals", height=500)
        st.plotly_chart(fig)




# Medal Tally
if user_menu == 'Medal Tally':
    st.sidebar.header("Medal Tally")
    years, country = helper.country_year_list(df)
    selected_year = st.sidebar.selectbox("Select Year", years)
    selected_country = st.sidebar.selectbox("Select Country", country)

    medal_tally = helper.fetch_medal_tally(df, selected_year, selected_country)
    if selected_year == 'Overall' and selected_country == 'Overall':
        st.title("Overall Tally")
    elif selected_year != 'Overall' and selected_country == 'Overall':
        st.title(f"Medal Tally in {selected_year} Olympics")
    elif selected_year == 'Overall' and selected_country != 'Overall':
        st.title(f"{selected_country} overall performance")
    else:
        st.title(f"{selected_country} performance in {selected_year} Olympics")

    # ✅ Add serial numbers starting from 1
    medal_tally_reset = medal_tally.reset_index(drop=True)
    medal_tally_reset.index += 1  # Start index from 1
    medal_tally_reset.index.name = "S.No"

    # ✅ Display with serial numbers
    st.table(medal_tally_reset)

# Overall Analysis
if user_menu == 'Overall Analysis':
    editions = df['Year'].nunique() - 1
    cities = df['City'].nunique()
    sports = df['Sport'].nunique()
    events = df['Event'].nunique()
    athletes = df['Name'].nunique()
    nations = df['region'].nunique()

    st.title("Top Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Editions")
        st.title(editions)
    with col2:
        st.header("Hosts")
        st.title(cities)
    with col3:
        st.header("Sports")
        st.title(sports)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Events")
        st.title(events)
    with col2:
        st.header("Nations")
        st.title(nations)
    with col3:
        st.header("Athletes")
        st.title(athletes)

    nations_over_time = helper.data_over_time(df, 'region')
    fig = px.line(nations_over_time, x="Edition", y="region")
    st.title("Participating Nations over the years")
    st.plotly_chart(fig)

    events_over_time = helper.data_over_time(df, 'Event')
    fig = px.line(events_over_time, x="Edition", y="Event")
    st.title("Events over the years")
    st.plotly_chart(fig)

    athlete_over_time = helper.data_over_time(df, 'Name')
    fig = px.line(athlete_over_time, x="Edition", y="Name")
    st.title("Athletes over the years")
    st.plotly_chart(fig)

    st.title("No. of Events over time (Every Sport)")
    fig, ax = plt.subplots(figsize=(20, 20))
    temp = df.drop_duplicates(['Year', 'Sport', 'Event'])
    ax = sns.heatmap(temp.pivot_table(index='Sport', columns='Year', values='Event', aggfunc='count').fillna(0).astype(int), annot=True)
    st.pyplot(fig)

    st.title("Most Successful Athletes")
    sport_list = df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0, 'Overall')
    selected_sport = st.selectbox('Select a Sport', sport_list)

    x = helper.most_successful(df, selected_sport)
    x.index = x.index + 1  # Start index from 1
    x.index.name = "Rank"  # Optional: Rename index column to "Rank"
    st.table(x)

# Country-wise Analysis
if user_menu == 'Country-wise Analysis':
    st.sidebar.header('Country-wise Analysis')

    # Get list of countries
    country_list = df['region'].dropna().unique().tolist()
    country_list.sort()
    selected_country = st.sidebar.selectbox('Select a Country', country_list)

    # Medal Tally over years
    country_df = helper.yearwise_medal_tally(df, selected_country)
    fig = px.line(country_df, x="Year", y="Medal")
    st.title(f"{selected_country} Medal Tally Over the Years")
    st.plotly_chart(fig)

    # Heatmap of sport vs year
    st.title(f"{selected_country} Excels in the Following Sports")
    pt = helper.country_event_heatmap(df, selected_country)

    if not pt.empty and pt.shape[0] > 0 and pt.shape[1] > 0:
        fig, ax = plt.subplots(figsize=(20, 20))
        ax = sns.heatmap(pt, annot=True, fmt='g', cmap='YlGnBu', linewidths=0.5, linecolor='gray')
        st.pyplot(fig)
    else:
        st.warning(f"⚠️ No sufficient event-level medal data available for {selected_country} to display a heatmap.")

    # Top 10 athletes
    st.title(f"Top 10 Athletes of {selected_country}")
    top10_df = helper.most_successful_countrywise(df, selected_country)
    top10_df.index = top10_df.index + 1       # Start index from 1
    top10_df.index.name = "Rank"              # Optional: Label index column as 'Rank'
    st.table(top10_df)


# Athlete-wise Analysis
if user_menu == 'Athlete wise Analysis':
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])

    x1 = athlete_df['Age'].dropna()
    x2 = athlete_df[athlete_df['Medal'] == 'Gold']['Age'].dropna()
    x3 = athlete_df[athlete_df['Medal'] == 'Silver']['Age'].dropna()
    x4 = athlete_df[athlete_df['Medal'] == 'Bronze']['Age'].dropna()

    fig = ff.create_distplot([x1, x2, x3, x4],
                             ['Overall Age', 'Gold Medalist', 'Silver Medalist', 'Bronze Medalist'],
                             show_hist=False, show_rug=False)
    fig.update_layout(autosize=False, width=1000, height=600)
    st.title("Distribution of Age")
    st.plotly_chart(fig)

    famous_sports = ['Basketball', 'Football', 'Tug-Of-War', 'Athletics', 'Swimming', 'Badminton',
                     'Gymnastics', 'Handball', 'Weightlifting', 'Wrestling', 'Water Polo', 'Hockey',
                     'Rowing', 'Fencing', 'Shooting', 'Boxing', 'Taekwondo', 'Cycling', 'Diving', 'Canoeing',
                     'Tennis', 'Golf', 'Softball', 'Archery', 'Volleyball', 'Table Tennis', 'Baseball', 'Rugby',
                     'Ice Hockey']

    x = []
    name = []
    for sport in famous_sports:
        ages = athlete_df[(athlete_df['Sport'] == sport) & (athlete_df['Medal'] == 'Gold')]['Age'].dropna()
        if not ages.empty:
            x.append(ages)
            name.append(sport)

    if x:
        fig = ff.create_distplot(x, name, show_hist=False, show_rug=False)
        fig.update_layout(autosize=False, width=1000, height=600)
        st.title("Distribution of Age wrt Sports (Gold Medalist)")
        st.plotly_chart(fig)
    else:
        st.warning("No valid gold medal age data available for the selected sports.")

    st.title('Height Vs Weight')

    # Define sport_list before using it
    sport_list = df['Sport'].dropna().unique().tolist()
    sport_list.sort()
    sport_list.insert(0, 'Overall')

    selected_sport = st.selectbox('Select a Sport', sport_list)
    temp_df = helper.weight_v_height(df, selected_sport)

    fig, ax = plt.subplots()
    ax = sns.scatterplot(x=temp_df['Weight'], y=temp_df['Height'], hue=temp_df['Medal'], style=temp_df['Sex'], s=60)
    st.pyplot(fig)


# Sport Video
# Feature 16: Sport Video (New Feature for Animated Video Link)
elif user_menu == "Sport Video":
    # Select sport
    sport_list = df['Sport'].dropna().unique().tolist()
    sport_list.sort()
    selected_sport = st.selectbox("Select Sport", sport_list)

    # Displaying the animated video
    st.title(f"Animated Video for {selected_sport}")

    # Check if the selected sport has a video link
    if selected_sport in sport_video_dict:
        video_link = sport_video_dict[selected_sport]
        st.video(video_link)
    else:
        st.warning(f"Sorry, no video link available for {selected_sport} yet.")


#medal predictor
if user_menu == 'Medal Predictor':
    st.title("🎯 Medal Predictor (Linear Regression)")

    predict_country = st.selectbox("Select a Country to Predict Medals", df['region'].dropna().unique())

    medal_df = df.dropna(subset=['Medal'])
    medal_df = medal_df.drop_duplicates(['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'])
    country_df = medal_df[medal_df['region'] == predict_country]
    medals_per_year = country_df.groupby('Year')['Medal'].count().reset_index()

    if medals_per_year.shape[0] > 1:
        X = medals_per_year[['Year']]
        y = medals_per_year['Medal']
        model = LinearRegression()
        model.fit(X, y)

        # Generate future Olympic years (every 4 years from 2024 to 2040)
        olympic_years = list(range(2028, 2096, 4))
        future_year = st.selectbox('Select Future Olympic Year', olympic_years)

        pred = model.predict([[future_year]])
        st.success(f"🏅 Predicted Medals for {predict_country} in {future_year}: **{int(pred[0])}** medals")

        fig = px.scatter(medals_per_year, x='Year', y='Medal', title='Historical Medal Count')
        fig.add_scatter(x=[future_year], y=[int(pred[0])], mode='markers+text',
                        text=[f"Predicted: {int(pred[0])}"], textposition='top center',
                        marker=dict(color='red', size=10))
        st.plotly_chart(fig)
    else:
        st.warning("Not enough historical data to predict medals.")



