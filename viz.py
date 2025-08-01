import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

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
    '250':    {'W': 250,  'F': 150, 'SF': 90, 'QF': 45,  'R16': 20},
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

# Initialize adjustment tables (multipliers, additions, subtractions)
types = default_matrix.index.tolist()
if 'multipliers' not in st.session_state:
    st.session_state['multipliers'] = pd.DataFrame([[1.0] * len(types)], columns=types, index=['Multiplier'])
if 'additions' not in st.session_state:
    st.session_state['additions'] = pd.DataFrame([[0.0] * len(types)], columns=types, index=['Addition'])
if 'subtractions' not in st.session_state:
    st.session_state['subtractions'] = pd.DataFrame([[0.0] * len(types)], columns=types, index=['Subtraction'])

# --- Tournament Adjustments Editor ---
st.subheader("🏷️ Tournament Multipliers")
multipliers = st.data_editor(
    st.session_state['multipliers'],
    num_rows="fixed",
    key="multipliers_editor",
    use_container_width=True
)
st.session_state['multipliers'] = multipliers.copy()

st.subheader("➕ Tournament Additions")
additions = st.data_editor(
    st.session_state['additions'],
    num_rows="fixed",
    key="additions_editor",
    use_container_width=True
)
st.session_state['additions'] = additions.copy()

st.subheader("➖ Tournament Subtractions")
subtractions = st.data_editor(
    st.session_state['subtractions'],
    num_rows="fixed",
    key="subtractions_editor",
    use_container_width=True
)
st.session_state['subtractions'] = subtractions.copy()

# --- Apply Adjustments to Scoring Matrix ---
if st.button("🔧 Apply Adjustments"):
    # Start from the original default matrix
    matrix = default_matrix.copy()
    # Apply multiplier, addition, subtraction row-wise
    matrix = matrix.mul(st.session_state['multipliers'].iloc[0], axis=0)
    matrix = matrix.add(st.session_state['additions'].iloc[0], axis=0)
    matrix = matrix.sub(st.session_state['subtractions'].iloc[0], axis=0)
    st.session_state['scoring_matrix'] = matrix
    st.session_state['matrix_editor_key'] = f"matrix_{pd.Timestamp.now().timestamp()}"

# --- Matrix Controls ---
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🔁 Reset to Default Matrix"):
        st.session_state['scoring_matrix'] = default_matrix.copy()
        st.session_state['multipliers'] = pd.DataFrame([[1.0] * len(types)], columns=types, index=['Multiplier'])
        st.session_state['additions'] = pd.DataFrame([[0.0] * len(types)], columns=types, index=['Addition'])
        st.session_state['subtractions'] = pd.DataFrame([[0.0] * len(types)], columns=types, index=['Subtraction'])
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

# --- Apply Scoring Model ---
if uploaded_file and st.button("Apply Scoring Model", key="apply_model"):
    st.session_state['scoring_matrix'] = scoring_matrix.copy()
    df = pd.read_csv(uploaded_file)

    # Calculate points using status quo and model matrices
    df['status_quo_points'] = df.apply(
        lambda row: scoring_matrix_dict.get(row['trntype_pts'], {}).get(row['round'], 0), axis=1
    )
    df['model_points'] = df.apply(
        lambda row: st.session_state['scoring_matrix'].loc[row['trntype_pts'], row['round']]
                    if row['trntype_pts'] in st.session_state['scoring_matrix'].index and
                       row['round'] in st.session_state['scoring_matrix'].columns else 0,
        axis=1
    )

    def summarize(col):
        return df.groupby('plyrnum')[col].sum().reset_index().rename(columns={col: col + '_total'})
    status_df = summarize('status_quo_points')
    model_df  = summarize('model_points')
    names_df  = df[['plyrnum', 'firstnam', 'lastnam']].drop_duplicates()

    combined = (
        names_df
        .merge(status_df, on='plyrnum', how='left')
        .merge(model_df, on='plyrnum', how='left')
    )

    combined['status_quo_rank'] = combined['status_quo_points_total'].rank(method='dense', ascending=False).fillna(0).astype(int)
    combined['model_rank']      = combined['model_points_total'].rank(method='dense', ascending=False).fillna(0).astype(int)
    combined = combined.dropna(subset=['firstnam', 'lastnam'])
    combined = combined[~combined['firstnam'].str.lower().eq('none')]
    combined = combined[~combined['lastnam'].str.lower().eq('none')]
    combined = combined[combined['status_quo_points_total'] >= 150]

    combined = combined.rename(columns={
        'plyrnum': 'player number',
        'firstnam':'first name',
        'lastnam':'last name',
        'status_quo_points_total':'status quo points',
        'model_points_total':'model points',
        'status_quo_rank':'status quo rank',
        'model_rank':'model rank'
    })
    combined = combined[
        ['player number','first name','last name',
         'status quo rank','status quo points',
         'model rank','model points']
    ].sort_values('model rank')
    combined['rank_change']    = combined['status quo rank'] - combined['model rank']
    combined['abs_change']     = combined['rank_change'].abs()
    combined['points_change']  = combined['model points'] - combined['status quo points']

    st.session_state['combined'] = combined

# --- Display Results if Available ---
if 'combined' in st.session_state:
    combined = st.session_state['combined']

    st.subheader("📊 Player Rankings Comparison")
    st.dataframe(combined, use_container_width=True)
    st.download_button(
        label="Download Results as CSV",
        data=combined.to_csv(index=False).encode('utf-8'),
        file_name="ranking_comparison.csv",
        mime='text/csv'
    )

    st.divider()
    st.subheader("📈 Top 10 Biggest Rank Movers")
    top = combined.sort_values('abs_change', ascending=False).head(10)
    st.dataframe(top[['player number','first name','last name','status quo rank','model rank','rank_change']], use_container_width=True)

    st.divider()
    # --- Rank Range Editor ---
    if 'rank_range_matrix' not in st.session_state:
        st.session_state['rank_range_matrix'] = pd.DataFrame({'From Rank':[1,11,21,31,41],'To Rank':[10,20,30,40,50]})
    ranges = st.data_editor(
        st.session_state['rank_range_matrix'],
        key="rank_range_editor",
        use_container_width=True
    )
    st.session_state['rank_range_matrix'] = ranges

    if st.button("📊 Apply Rank Range Analysis", key="apply_range"):
        results = []
        for _, row in ranges.iterrows():
            fr, to = int(row['From Rank']), int(row['To Rank'])
            subset = combined[(combined['status quo rank'] >= fr) & (combined['status quo rank'] <= to)]
            results.append({
                'From Rank': fr,
                'To Rank': to,
                'Status Quo Total': subset['status quo points'].sum(),
                'Model Total': subset['model points'].sum(),
                'Delta': subset['model points'].sum() - subset['status quo points'].sum()
            })
        results_df = pd.DataFrame(results)
        st.subheader("🧮 Rank Range Totals")
        st.dataframe(results_df, use_container_width=True)
        st.download_button(
            label="📥 Download Rank Range Totals as CSV",
            data=results_df.to_csv(index=False).encode('utf-8'),
            file_name="rank_range_totals.csv",
            mime='text/csv'
        )

    st.divider()
    # --- Points Comparison Chart ---
    min_val = min(combined['status quo points'].min(), combined['model points'].min())
    max_val = max(combined['status quo points'].max(), combined['model points'].max())
    line_df = pd.DataFrame({'x':[min_val, max_val],'y':[min_val, max_val]})
    line = alt.Chart(line_df).mark_line(color='gray', strokeDash=[5,5]).encode(x='x', y='y')
    scatter = alt.Chart(combined).mark_circle(size=80).encode(
        x=alt.X('status quo points', title='Status Quo Points', scale=alt.Scale(domain=[min_val,max_val])),
        y=alt.Y('model points', title='Model Points', scale=alt.Scale(domain=[min_val,max_val])),
        color=alt.Color('points_change', scale=alt.Scale(scheme='redblue', domainMid=0), title='Points Change'),
        tooltip=['player number','first name','last name','status quo rank','model rank','rank_change','status quo points','model points','points_change']
    ).interactive().properties(width=600, height=600)
    st.subheader("🔁 Points Comparison: Status Quo vs Model")
    st.altair_chart(line + scatter, use_container_width=True)
else:
    st.info("Please upload your cleaned match-level CSV file to begin.")
