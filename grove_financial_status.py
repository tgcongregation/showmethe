import numpy as np
import pandas as pd
import streamlit as st
from datetime import date, timedelta
from calendar import monthrange
import os
import glob
from dateutil.relativedelta import relativedelta

budget_by_year = {}
heading_button_spacing = [7, 1, 1]

# Streamlit app
def main():
    st.set_page_config(page_title="Grove Financial Status", layout="wide", initial_sidebar_state="auto",
                       menu_items=None)
    st.markdown("""
    <style>
        .dataframe {
            width: 100%;
            overflow-x: auto;
        }
        .dataframe td, .dataframe th {
            white-space: nowrap;
            text-align: left;
        }
    </style>
    """, unsafe_allow_html=True)

    transactions = None

    try:
        pattern = os.path.join('data', 'transactions_[0-9][0-9][0-9][0-9].csv')
        transaction_files = glob.glob(pattern)

        if not transaction_files:
            st.warning("No transaction files found matching the pattern 'transaction_yyyy'")
            return

        for file_path in transaction_files:
            transactions = load_data(file_path, transactions)

        if transactions is not None:
            transactions = transactions.sort_values(by="transaction_date")

    except Exception as e:
        st.error(f"An error occurred during initial file loading: {e}")

    if transactions is not None:
        prep_budget_detail_info(transactions)

        monthly_tab, quarterly_tab, yearly_tab, budget_tab, raw_data_tab = st.tabs(
            ["Monthly", "Quarterly", "Yearly", "Budget", "Raw Data"])

        with monthly_tab:
            establish_monthly_tab(transactions)

        with quarterly_tab:
            establish_quarterly_tab(transactions)

        with budget_tab:
            establish_budget_tab()

        with yearly_tab:
            establish_yearly_tab(transactions)

        with raw_data_tab:
            st.header("Raw Data")
            st.write("### DataFrame from the CSV file:")
            st.dataframe(transactions, hide_index=True, width=None)


def prep_budget_detail_info(transactions):
    transactions['remaining_budget'] = transactions['budget_amount']
    transactions['transaction_amount'] = transactions['credit'] - transactions['debit']

    sign = np.where(transactions['income'], -1, 1)
    transactions['remaining_budget'] += sign * transactions.groupby(['year', 'category'])[
        'transaction_amount'].transform('cumsum')

    transactions['percent_remaining'] = (
        (transactions['remaining_budget']
         .div(transactions['budget_amount'])
         .mul(100)
         .round(2)
         .replace([np.inf, -np.inf], '')
         .fillna('')
         .apply(lambda x: f'{x}%' if x != '' else '')
         )
    )


def load_data(file_path, transactions):
    try:
        year = os.path.basename(file_path).split('_')[1][:4]
        temp_transactions = load_and_prepare_csv(file_path)
        temp_transactions['year'] = year
        budget_file = os.path.join('data', f'budget_{year}.csv')

        if os.path.exists(budget_file):
            try:
                if year not in budget_by_year:
                    budget_by_year[year] = prep_budget_from_file(budget_file, year)
                temp_transactions = temp_transactions.merge(
                    budget_by_year[year],
                    on=['category', 'year'],
                    how='left'
                )
                transactions = temp_transactions if transactions is None else pd.concat(
                    [transactions, temp_transactions])
            except Exception as e:
                st.warning(f"Error loading budget file {budget_file}: {e}")
        else:
            st.warning(f"Budget file not found for year {year}")
    except Exception as e:
        st.error(f"Error processing file {file_path}: {e}")
    return transactions


def prep_budget_from_file(budget_file, year):
    temp_budget = pd.read_csv(budget_file)
    temp_budget = temp_budget.rename(columns={
        'level1': 'level1_budget',
        'level2': 'level2_budget',
        'level3': 'level3_budget',
        'amount': 'budget_amount',
        'income': 'income'
    })
    temp_budget['category'] = temp_budget.apply(
        lambda x: ':'.join(str(val) for val in [x['level1_budget'], x['level2_budget'], x['level3_budget']]
                           if pd.notna(val)),
        axis=1
    )

    temp_budget['year'] = year
    return temp_budget


def establish_yearly_tab(df):
    st.header("Yearly Data")
    col1, col2, col3 = st.columns(heading_button_spacing)
    # Initialize Streamlit state for the current quarter's start and end dates
    if st.session_state.get('current_year_start') is None:
        st.session_state['current_year_start'], st.session_state[
            'current_year_end'] = get_yearly_boundaries(
            date.today()
        )
    with col2:
        if st.button("Previous", "yearly_prev"):
            st.session_state['current_year_start'], st.session_state[
                'current_year_end'] = get_yearly_boundaries(
                st.session_state['current_year_start'] + relativedelta(months=-12, day=1)
            )
    with col3:
        if st.button("Next", "yearly_next"):
            st.session_state['current_year_start'], st.session_state[
                'current_year_end'] = get_yearly_boundaries(
                st.session_state['current_year_end'] + relativedelta(months=12, day=1)
            )
    with col1:
        st.write(f"{st.session_state['current_year_start'].year}")
    filtered_df = filter_temporally(df, st.session_state['current_year_start'],
                                    st.session_state['current_year_end'])
    level1_level2_summary = establish_level1_level2_summary_table(filtered_df)
    level1_summary = establish_level1_summary_table(filtered_df)

    filtered_df = extract_essential_columns(filtered_df)
    st.dataframe(filtered_df, hide_index=True, width=None)

    display_level_tables(level1_level2_summary, level1_summary)


def establish_budget_tab():
    st.header("Budget Data")
    col1, col2, col3 = st.columns(heading_button_spacing)
    # Initialize Streamlit state for the current budget's start and end dates
    if st.session_state.get('current_budget_year') is None:
        st.session_state['current_budget_year'] = date.today().year
    with col2:
        if st.button("Previous", "budget_prev"):
            st.session_state['current_budget_year'] = st.session_state['current_budget_year'] - 1
    with col3:
        if st.button("Next", "budget_next"):
            st.session_state['current_budget_year'] = st.session_state['current_budget_year'] + 1
    with col1:
        st.write(f"{st.session_state['current_budget_year']}")
    filtered_df = budget_by_year[str(st.session_state['current_budget_year'])]

    level1_level2_summary  = establish_level1_level2_budget_table(filtered_df)
    level1_summary  = establish_level1_budget_table(filtered_df)

    filtered_df = filtered_df[['category', 'budget_amount', 'income', 'level1_budget', 'level2_budget', 'level3_budget']].rename(columns={
        'category': 'Category',
        'budget_amount': 'Budget Amount',
        'income': 'Income',
        'level1_budget': 'Level 1',
        'level2_budget': 'Level 2',
        'level3_budget': 'Level 3'
    })
    st.dataframe(filtered_df, hide_index=True, width=None)

    st.subheader("Level 1 and Level 2 Summary")
    st.dataframe(level1_level2_summary, hide_index=True, width=None)
    st.subheader("Level 1 Summary")
    st.dataframe(level1_summary, hide_index=True, width=None)

def establish_quarterly_tab(df):
    st.header("Quarterly Data")
    col1, col2, col3 = st.columns(heading_button_spacing)
    # Initialize Streamlit state for the current quarter's start and end dates
    if st.session_state.get('current_quarter_start') is None:
        st.session_state['current_quarter_start'], st.session_state[
            'current_quarter_end'] = get_quarter_boundaries(
            date.today()
        )
    with col2:
        if st.button("Previous", "quarter_prev"):
            st.session_state['current_quarter_start'], st.session_state[
                'current_quarter_end'] = get_quarter_boundaries(
                st.session_state['current_quarter_start'] + relativedelta(months=-3, day=1)
            )
    with col3:
        if st.button("Next", "quarter_next"):
            st.session_state['current_quarter_start'], st.session_state[
                'current_quarter_end'] = get_quarter_boundaries(
                st.session_state['current_quarter_end'] + relativedelta(months=3, day=1)
            )
    with col1:
        current_quarter = get_current_quarter(st.session_state['current_quarter_start'])
        st.write(f"{current_quarter} Quarter, {st.session_state['current_quarter_start'].year}")
    filtered_df = filter_temporally(df, st.session_state['current_quarter_start'],
                                    st.session_state['current_quarter_end'])
    level1_level2_summary = establish_level1_level2_summary_table(filtered_df)
    level1_summary = establish_level1_summary_table(filtered_df)

    filtered_df = extract_essential_columns(filtered_df)
    st.dataframe(filtered_df, hide_index=True, width=None)

    display_level_tables(level1_level2_summary, level1_summary)

def establish_monthly_tab(df):
    # Optionally display the filtered DataFrame for the Monthly tab
    st.header("Monthly Data")
    col1, col2, col3 = st.columns(heading_button_spacing)
    # Initialize Streamlit state for the current month's start and end dates
    if st.session_state.get('current_month_start') is None:
        st.session_state['current_month_start'], st.session_state['current_month_end'] = get_month_boundaries(
            date.today()
        )
    with col2:
        if st.button("Previous", "month_prev"):
            st.session_state['current_month_start'], st.session_state[
                'current_month_end'] = get_month_boundaries(
                st.session_state['current_month_start'] + relativedelta(months=-1, day=1)
            )
    with col3:
        if st.button("Next", "month_next"):
            st.session_state['current_month_start'], st.session_state[
                'current_month_end'] = get_month_boundaries(
                st.session_state['current_month_end'] + relativedelta(months=1, day=1)
            )
    with col1:
        st.write(f"{st.session_state['current_month_start'].strftime('%B %Y')}")
    filtered_df = filter_temporally(df, st.session_state['current_month_start'],
                                    st.session_state['current_month_end'])
    level1_level2_summary = establish_level1_level2_summary_table(filtered_df)
    level1_summary = establish_level1_summary_table(filtered_df)

    filtered_df = extract_essential_columns(filtered_df)
    st.dataframe(filtered_df, hide_index=True, width=None)

    display_level_tables(level1_level2_summary, level1_summary)


def display_level_tables(level1_level2_summary, level1_summary):
    # Now add Level 2 budget summary
    st.subheader("Level 1 & 2 Summary")
    st.dataframe(level1_level2_summary, hide_index=True, width=None)
    # Now add Level 1 budget summary
    st.subheader("Level 1 Summary")
    st.dataframe(level1_summary, hide_index=True, width=None)


def extract_essential_columns(filtered_df):
    filtered_df = filtered_df[[
        'category',
        'transaction_date',
        'description',
        'transaction_amount',
        'budget_amount',
        'remaining_budget',
        'percent_remaining'
    ]]
    filtered_df = filtered_df.rename(columns={
        'category': 'Category',
        'transaction_date': 'Transaction Date',
        'description': 'Description',
        'transaction_amount': 'Transaction Amount',
        'budget_amount': 'Budget Amount',
        'remaining_budget': 'Remaining Budget',
        'percent_remaining': 'Percent Remaining'
    })
    return filtered_df


def filter_temporally(df, start_date, end_date):
    # Filter DataFrame based on transaction_date within the current month's range
    filtered_df = df[
        (df['transaction_date'] >= start_date) &
        (df['transaction_date'] <= end_date)
        ]
    return filtered_df


def load_and_prepare_csv(file_path):
    df = pd.read_csv(file_path)
    df[['level1', 'level2', 'level3']] = df['category'].str.split(':', n=2, expand=True)
    reordered_columns = (['category', 'level1', 'level2', 'level3'] +
                         [col for col in df.columns
                          if col not in ['category', 'level1', 'level2', 'level3']])
    df = df[reordered_columns]
    df.fillna("", inplace=True)
    df.drop(columns=['balance'], inplace=True)

    # Convert to datetime and then extract just the date component
    df['transaction_date'] = pd.to_datetime(df['transaction_date']).dt.date
    return df


def get_month_boundaries(reference_date: date) -> tuple[date, date]:
    """
    Returns the start and end dates for the month of the given reference date.
    """
    year = reference_date.year
    month = reference_date.month
    _, last_day = monthrange(year, month)

    # Create start and end dates without time component
    start_date = date(year, month, 1)
    end_date = date(year, month, last_day)

    return start_date, end_date


def get_quarter_boundaries(reference_date):
    quarter = (reference_date.month - 1) // 3
    quarter_start = date(reference_date.year, quarter * 3 + 1, 1)

    if quarter == 3:  # Q4
        next_year = date(reference_date.year + 1, 1, 1)
        quarter_end = next_year - timedelta(days=1)
    else:
        next_quarter = date(reference_date.year, (quarter + 1) * 3 + 1, 1)
        quarter_end = next_quarter - timedelta(days=1)

    return quarter_start, quarter_end


def get_current_quarter(reference_date):
    current_month = reference_date.month

    if 1 <= current_month <= 3:
        return "1st"
    elif 4 <= current_month <= 6:
        return "2nd"
    elif 7 <= current_month <= 9:
        return "3rd"
    else:
        return "4th"


def get_yearly_boundaries(reference_date):
    year_start = date(reference_date.year, 1, 1)
    year_end = date(reference_date.year, 12, 31)
    return year_start, year_end


def establish_level1_level2_summary_table(filtered_df):
    summary_df = (
        filtered_df.groupby(['level1', 'level2'], as_index=False)
        .agg({'debit': 'sum', 'credit': 'sum', 'transaction_amount': 'sum'})
        .rename(columns={
            'level1': 'Level 1',
            'level2': 'Level 2',
            'debit': 'Total Debit',
            'credit': 'Total Credit',
            'transaction_amount': 'Total Transactions'
        })
    )
    return summary_df

def establish_level1_summary_table(filtered_df):
    summary_df = (
        filtered_df.groupby(['level1'], as_index=False)
        .agg({'debit': 'sum', 'credit': 'sum', 'transaction_amount': 'sum'})
        .rename(columns={
            'level1': 'Level 1',
            'debit': 'Total Debit',
            'credit': 'Total Credit',
            'transaction_amount': 'Total Transactions'
        })
    )
    return summary_df

def establish_level1_level2_budget_table(filtered_df):
    summary_df = (
        filtered_df.groupby(['level1_budget', 'level2_budget'], as_index=False)
        .agg({'budget_amount': 'sum'})
        .rename(columns={
            'level1_budget': 'Level 1',
            'level2_budget': 'Level 2',
            'budget_amount': 'Budget Amount'
        })
    )
    return summary_df

def establish_level1_budget_table(filtered_df):
    summary_df = (
        filtered_df.groupby(['level1_budget', 'income'], as_index=False)
        .agg({'budget_amount': 'sum'})
    )

    expense_total = summary_df[~summary_df['income']]['budget_amount'].sum()
    expense_row = pd.DataFrame({
        'level1_budget': ['Expense Total'],
        'budget_amount': [expense_total]
    })
    income_total = summary_df[summary_df['income']]['budget_amount'].sum()
    income_row = pd.DataFrame({
        'level1_budget': ['Income Total'],
        'budget_amount': [income_total]
    })
    summary_df = pd.concat([summary_df, income_row, expense_row], ignore_index=True)
    summary_df = summary_df[['level1_budget', 'budget_amount', 'income']].rename(columns={
        'level1_budget': 'Level 1',
        'budget_amount': 'Budget Amount',
        'income': 'Income'
    })

    return summary_df


# Run the app
if __name__ == '__main__':
    main()
