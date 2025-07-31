import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="ATP Doubles Ranker", layout="wide")
st.title("🎾 ATP Doubles Ranking Simulator")
st.divider()

# --- File Upload ---
uploaded_file = st.file_uploader("Upload match-level CSV (with status_quo column)", type="csv")
st.divider()
# --- Define static scoring matrix ---
scoring_matrix_dict = {
    'GS':     {'W': 2000, 'F': 1200, 'SF': 720, 'QF': 360, 'R16': 180, 'R32': 90, 'R64': 45},
    '1000':   {'W': 1000, 'F': 600, 'SF': 360, 'QF': 180, 'R16': 90, 'R32': 45},
    '500':    {'W': 500,  'F': 300, 'SF': 180, 'QF': 90,  'R16': 45},
    '250':    {'W': 250,  'F': 150, 'SF': 90,  'QF': 45,  'R16': 20},
    'CH 175': {'W': 175,  'F': 100, 'SF': 60,  'QF': 32,  'R16': 15},
    'CH 125': {'W': 125,  'F': 75, 'SF': 45,  'QF': 25,  'R16': 11},
    'CH 100': {'W': 100,  'F': 60,  'SF': 36,  'QF': 20,  'R16': 9},
    'CH 75':  {'W': 75,   'F': 50,  'SF': 30,  'QF': 16,  'R16': 7},
    'CH 50':  {'W': 50,   'F': 30,  'SF': 17,  'QF': 9,   'R16': 4},
    'ITF 25': {'W': 25,   'F': 16,  'SF': 8,   'QF': 3,   'R16': 1},
    'ITF 15': {'W': 15,   'F': 8,   'SF': 4,   'QF': 2,   'R16': 1}
}
default_matrix = pd.DataFrame.from_dict(scoring_matrix_dict, orient='index').fillna(0)

# --- Init session state ---
if 'scoring_matrix' not in st.session_state:
    st.session_state['scoring_matrix'] = default_matrix.copy()

if 'matrix_editor_key' not in st.session_state:
    st.session_state['matrix_editor_key'] = 'matrix_v1'

# --- Matrix Controls ---
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🔁 Reset to Default Matrix"):
        st.session_state['scoring_matrix'] = default_matrix.copy()
        st.session_state['matrix_editor_key'] = f"matrix_{pd.Timestamp.now().timestamp()}"

with col2:
    st.download_button(
        label="📥 Download Scoring Matrix as CSV",
        data=st.session_state['scoring_matrix'].to_csv().encode('utf-8'),
        file_name="current_scoring_matrix.csv",
        mime="text/csv"
    )

# --- Matrix Editor ---
st.subheader("Edit Scoring Matrix")
scoring_matrix = st.data_editor(
    st.session_state['scoring_matrix'],
    num_rows="dynamic",
    key=st.session_state['matrix_editor_key'],
    use_container_width=True
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if st.button("Apply Scoring Model"):
        # Store edited matrix
        st.session_state['scoring_matrix'] = scoring_matrix.copy()

        # --- Points Assignment ---
        def get_status_points(row):
            try:
                return scoring_matrix_dict[row['trntype_pts']][row['round']]
            except:
                return 0

        def get_model_points(row):
            try:
                return scoring_matrix.loc[row['trntype_pts'], row['round']]
            except:
                return 0

        df['status_quo_points'] = df.apply(get_status_points, axis=1)
        df['model_points'] = df.apply(get_model_points, axis=1)

        # --- Aggregation ---
        def summarize_points(colname):
            return (
                df.groupby('plyrnum')[colname]
                .sum()
                .reset_index()
                .rename(columns={colname: colname + '_total'})
            )

        status_df = summarize_points('status_quo_points')
        model_df = summarize_points('model_points')
        names_df = df[['plyrnum', 'firstnam', 'lastnam']].drop_duplicates()

        combined = (
            names_df
            .merge(status_df, on='plyrnum', how='left')
            .merge(model_df, on='plyrnum', how='left')
        )

        # --- Rankings ---
        combined['status_quo_rank'] = (
            combined['status_quo_points_total']
            .rank(method='dense', ascending=False)
            .fillna(0)
            .astype(int)
        )
        combined['model_rank'] = (
            combined['model_points_total']
            .rank(method='dense', ascending=False)
            .fillna(0)
            .astype(int)
        )

        # --- Filter ---
        combined = combined.dropna(subset=['firstnam', 'lastnam'])
        combined = combined[combined['firstnam'].str.lower() != 'none']
        combined = combined[combined['lastnam'].str.lower() != 'none']
        combined = combined[combined['status_quo_points_total'] >= 150]


        # --- Rename & Reorder ---
        combined = combined.rename(columns={
            'plyrnum': 'player number',
            'firstnam': 'first name',
            'lastnam': 'last name',
            'status_quo_points_total': 'status quo points',
            'model_points_total': 'model points',
            'status_quo_rank': 'status quo rank',
            'model_rank': 'model rank'
        })

        combined = combined[
            ['player number', 'first name', 'last name',
             'status quo rank', 'status quo points',
             'model rank', 'model points']
        ].sort_values(by='model rank')

        combined['rank_change'] = combined['status quo rank'] - combined['model rank']
        combined['abs_change'] = combined['rank_change'].abs()
        combined['points_change'] = combined['model points'] - combined['status quo points']


        # --- Output Table ---
        st.subheader("📊 Player Rankings Comparison ")
        st.dataframe(combined, use_container_width=True)

        st.download_button(
            label="Download Results as CSV",
            data=combined.to_csv(index=False).encode('utf-8'),
            file_name="ranking_comparison.csv",
            mime='text/csv'
        )


        st.divider()


        st.subheader("📈 Top 10 Biggest Rank Movers")
        top_movers = combined.sort_values(by='abs_change', ascending=False).head(10)
        st.dataframe(top_movers[
            ['player number', 'first name', 'last name',
            'status quo rank', 'model rank', 'rank_change']
        ], use_container_width=True)


        st.divider()

        st.subheader("🔀 Rank Comparison: Status Quo vs Model")

        scatter = alt.Chart(combined).mark_circle(size=80).encode(
            x=alt.X('status quo points', title='Status Quo Points'),
            y=alt.Y('model points', title='Model Points'),
            tooltip=[
                'player number', 'first name', 'last name',
                'status quo rank', 'model rank', 'rank_change',
                'status quo points', 'model points', 'points_change'
            ],
            color=alt.Color(
                'points_change',
                scale=alt.Scale(scheme='redblue', domainMid=0),
                title='Points Change'
            )
        ).properties(
            width=600,
            height=600
        ).interactive()

        st.altair_chart(scatter, use_container_width=True)


    #     # --- Histogram: Status Quo Points ---
    #     st.subheader("📈 Status Quo Points Distribution")
    #     hist_statusquo_df = pd.DataFrame({'points': combined['status quo points']})
    #     chart1 = alt.Chart(hist_statusquo_df).mark_bar().encode(
    #         alt.X("points", bin=alt.Bin(maxbins=30), title='Status Quo Points'),
    #         y=alt.Y('count()', title='Number of Players')
    #     ).properties(width=600, height=400)
    #     st.altair_chart(chart1, use_container_width=True)

    #     # --- Histogram + KDE: Model Points ---
    #     st.subheader("📉 Model Points Distribution")
    #     hist_model_df = pd.DataFrame({'points': combined['model points']})
    #     chart2 = alt.Chart(hist_model_df).mark_bar(opacity=0.6).encode(
    #         alt.X("points", bin=alt.Bin(maxbins=30), title='Model Points'),
    #         y=alt.Y('count()', title='Number of Players')
    #     )
    #     kde = alt.Chart(hist_model_df).transform_density(
    #         'points', as_=['points', 'density']
    #     ).mark_line(color='red').encode(
    #         x='points:Q',
    #         y='density:Q'
    #     )
    #     st.altair_chart(chart2 + kde, use_container_width=True)

    # else:
    #     st.info("Click 'Apply Scoring Model' to process rankings.")


else:
    st.info("Please upload your cleaned match-level CSV file to begin.")
