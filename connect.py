import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
import json
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import warnings
import pymongo
from pymongo import MongoClient
import hashlib
from config import *
import os
warnings.filterwarnings('ignore')

# MongoDB connection
def init_mongodb():
    """Initialize MongoDB connection - returns None if not available"""
    try:
        # Try to connect to MongoDB using config with very short timeout
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=2000, connectTimeoutMS=2000)
        # Test the connection
        client.admin.command('ping')
        db = client[MONGODB_DB_NAME]
        st.session_state.mongodb_available = True
        print("MongoDB connection successful")
        return db
    except Exception as e:
        st.session_state.mongodb_available = False
        st.warning("⚠️ Could not connect to MongoDB. Data will be stored only for this session.")
        return None

# User authentication functions
def hash_email(email):
    return hashlib.sha256(email.encode()).hexdigest()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate_user():
    if 'user_email' not in st.session_state:
        st.session_state.user_email = None
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.sidebar.subheader("🔐 Login or Register")
        email = st.sidebar.text_input("Enter your email:", key="login_email")
        password = st.sidebar.text_input("Enter your password:", type="password", key="login_password")
        db = st.session_state.get('db_connection', None)
        if st.sidebar.button("Login/Register"):
            if email and password:
                user_id = hash_email(email.lower().strip())
                user_email = email.lower().strip()
                user_doc = None
                if db is not None and st.session_state.get('mongodb_available', False):
                    user_doc = db.users.find_one({"user_id": user_id})
                if user_doc:
                    # User exists, check password
                    stored_hash = user_doc.get('password_hash', None)
                    if stored_hash and stored_hash == hash_password(password):
                        st.session_state.user_email = user_email
                        st.session_state.user_id = user_id
                        st.session_state.authenticated = True
                        st.rerun()
                    else:
                        st.sidebar.error("Incorrect password. Please try again.")
                else:
                    # New user registration
                    if db is not None and st.session_state.get('mongodb_available', False):
                        db.users.insert_one({
                            "user_id": user_id,
                            "email": user_email,
                            "password_hash": hash_password(password),
                            "expenses": [],
                            "income": [],
                            "budget": DEFAULT_BUDGET.copy(),
                            "savings_goals": DEFAULT_SAVINGS_GOALS.copy(),
                            "user_profile": {
                                'name': user_email.split('@')[0],
                                'email': user_email,
                                'member_since': datetime.now().strftime('%Y-%m-%d'),
                                'currency': '₹',
                                'language': 'English'
                            },
                            "last_updated": datetime.now()
                        })
                    st.session_state.user_email = user_email
                    st.session_state.user_id = user_id
                    st.session_state.authenticated = True
                    st.sidebar.success("Account created and logged in!")
                    st.rerun()
            else:
                st.sidebar.error("Please enter both email and password.")
        return False

    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state.user_email = None
        st.session_state.user_id = None
        st.session_state.authenticated = False
        for key in list(st.session_state.keys()):
            if key not in ['user_email', 'user_id', 'authenticated']:
                del st.session_state[key]
        st.rerun()

    st.sidebar.success(f"Logged in as: {st.session_state.user_email}")
    return True

# Database operations
def load_user_data(db, user_id):
    """Load user data from MongoDB or session state"""
    try:
        if db is not None and st.session_state.get('mongodb_available', False):
            # Try to load from MongoDB
            user_data = db.users.find_one({"user_id": user_id})
            if user_data:
                # Load expenses
                expenses_data = user_data.get('expenses', [])
                st.session_state.expenses = pd.DataFrame(expenses_data) if expenses_data else pd.DataFrame(columns=['date', 'category', 'amount', 'description'])
                # Ensure 'imported' column exists
                if 'imported' not in st.session_state.expenses.columns:
                    st.session_state.expenses['imported'] = False
                else:
                    st.session_state.expenses['imported'] = st.session_state.expenses['imported'].fillna(False)
                # Load income
                income_data = user_data.get('income', [])
                st.session_state.income = pd.DataFrame(income_data) if income_data else pd.DataFrame(columns=['date', 'source', 'amount', 'frequency'])
                # Ensure 'imported' column exists
                if 'imported' not in st.session_state.income.columns:
                    st.session_state.income['imported'] = False
                else:
                    st.session_state.income['imported'] = st.session_state.income['imported'].fillna(False)
                # Load budget
                st.session_state.budget = user_data.get('budget', DEFAULT_BUDGET.copy())
                # Load savings goals
                st.session_state.savings_goals = user_data.get('savings_goals', DEFAULT_SAVINGS_GOALS.copy())
                # Load user profile
                st.session_state.user_profile = user_data.get('user_profile', {
                    'name': st.session_state.user_email.split('@')[0],
                    'email': st.session_state.user_email,
                    'member_since': datetime.now().strftime('%Y-%m-%d'),
                    'currency': '₹',
                    'language': 'English'
                })
                print("Loading user data for user_id:", user_id)
                print("Loaded user data:", user_data)
            else:
                # Initialize new user with default data
                initialize_new_user()
        else:
            # MongoDB not available, use session state
            if 'expenses' not in st.session_state:
                initialize_new_user()
            # Ensure 'imported' column exists in session state
            if 'imported' not in st.session_state.expenses.columns:
                st.session_state.expenses['imported'] = False
            else:
                st.session_state.expenses['imported'] = st.session_state.expenses['imported'].fillna(False)
            if 'imported' not in st.session_state.income.columns:
                st.session_state.income['imported'] = False
            else:
                st.session_state.income['imported'] = st.session_state.income['imported'].fillna(False)
            print("Using session state storage (MongoDB not available)")
    except Exception as e:
        st.error(f"❌ Error loading user data from database. Using session storage.\nDetails: {e}")
        # Fallback to session state
        if 'expenses' not in st.session_state:
            initialize_new_user()
        # Ensure 'imported' column exists in session state
        if 'imported' not in st.session_state.expenses.columns:
            st.session_state.expenses['imported'] = False
        else:
            st.session_state.expenses['imported'] = st.session_state.expenses['imported'].fillna(False)
        if 'imported' not in st.session_state.income.columns:
            st.session_state.income['imported'] = False
        else:
            st.session_state.income['imported'] = st.session_state.income['imported'].fillna(False)
        print("Falling back to session state storage")

def save_user_data(db, user_id):
    """Save user data to MongoDB or session state"""
    try:
        # Convert DataFrames to list of dictionaries
        expenses_data = st.session_state.expenses.to_dict('records') if not st.session_state.expenses.empty else []
        income_data = st.session_state.income.to_dict('records') if not st.session_state.income.empty else []
        # Convert all date fields to string (ISO format)
        for expense in expenses_data:
            if isinstance(expense['date'], (datetime, )):
                expense['date'] = expense['date'].strftime('%Y-%m-%d')
            elif isinstance(expense['date'], (pd.Timestamp, )):
                expense['date'] = expense['date'].strftime('%Y-%m-%d')
            elif isinstance(expense['date'], (date, )):
                expense['date'] = expense['date'].isoformat()
        for income in income_data:
            if isinstance(income['date'], (datetime, )):
                income['date'] = income['date'].strftime('%Y-%m-%d')
            elif isinstance(income['date'], (pd.Timestamp, )):
                income['date'] = income['date'].strftime('%Y-%m-%d')
            elif isinstance(income['date'], (date, )):
                income['date'] = income['date'].isoformat()
        if db is not None and st.session_state.get('mongodb_available', False):
            # Save to MongoDB
            # Do not overwrite password_hash
            user_data = db.users.find_one({"user_id": user_id}) or {}
            password_hash = user_data.get('password_hash', None)
            new_data = {
                "user_id": user_id,
                "email": st.session_state.user_email,
                "expenses": expenses_data,
                "income": income_data,
                "budget": st.session_state.budget,
                "savings_goals": st.session_state.savings_goals,
                "user_profile": st.session_state.user_profile,
                "last_updated": datetime.now()
            }
            if password_hash:
                new_data['password_hash'] = password_hash
            db.users.replace_one({"user_id": user_id}, new_data, upsert=True)
            print("Saving user data to MongoDB:", new_data)
            print("user_id for saving:", st.session_state.user_id)
            print("Expenses to save:", expenses_data)
            print("Income to save:", income_data)
        else:
            # MongoDB not available, data is already in session state
            print("Data saved to session state (MongoDB not available)")
        return True
    except Exception as e:
        st.error(f"❌ Error saving user data to database. Data is only saved for this session.\nDetails: {e}")
        # Data is already in session state, so we don't need to show an error
        return True

def initialize_new_user():
    """Initialize session state for new user"""
    st.session_state.expenses = pd.DataFrame(columns=['date', 'category', 'amount', 'description'])
    st.session_state.income = pd.DataFrame(columns=['date', 'source', 'amount', 'frequency'])
    st.session_state.budget = DEFAULT_BUDGET.copy()
    st.session_state.savings_goals = DEFAULT_SAVINGS_GOALS.copy()
    st.session_state.user_profile = {
        'name': st.session_state.user_email.split('@')[0] if st.session_state.user_email else 'User',
        'email': st.session_state.user_email,
        'member_since': datetime.now().strftime('%Y-%m-%d'),
        'currency': '₹',
        'language': 'English'
    }

# Page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern theme and responsive CSS
st.markdown("""
<style>
    html, body, [class*='css']  { font-family: 'Inter', 'Segoe UI', Arial, sans-serif; }
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        letter-spacing: -1px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .success-card {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .warning-card {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
    }
    .danger-card {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    /* Responsive tables/charts */
    .element-container, .stDataFrame, .stTable, .stPlotlyChart {
        overflow-x: auto !important;
        max-width: 100vw;
    }
    @media (max-width: 900px) {
        .main-header { font-size: 1.5rem; }
        .metric-card { font-size: 1rem; }
        .stTabs [role='tablist'] { flex-wrap: wrap; }
    }
    @media (max-width: 600px) {
        .main-header { font-size: 1.1rem; }
        .metric-card { font-size: 0.9rem; }
        .stTabs [role='tablist'] { flex-direction: column; }
    }
    /* Hide vertical scrollbar for the main content and sidebar */
    ::-webkit-scrollbar {
        width: 0px;
        background: transparent;
    }
</style>
""", unsafe_allow_html=True)

# Constants are now imported from config.py

# Helper functions (modified to save to DB)
def add_expense(date, category, amount, description="", imported=False):
    new_expense = pd.DataFrame({
        'date': [date],
        'category': [category],
        'amount': [amount],
        'description': [description],
        'imported': [imported]
    })
    # Duplicate prevention logic (unchanged)
    tx_str = f"{date}|{category}|{amount}|{description}".lower()
    tx_hash = hashlib.sha256(tx_str.encode()).hexdigest()
    if 'expense_hashes' not in st.session_state:
        st.session_state.expense_hashes = set()
    if tx_hash in st.session_state.expense_hashes:
        st.warning("Duplicate expense detected. This transaction was not added.")
        return
    st.session_state.expense_hashes.add(tx_hash)
    st.session_state.expenses = pd.concat([st.session_state.expenses, new_expense], ignore_index=True)
    # Save to database (if available) - use existing connection
    if st.session_state.get('mongodb_available', False) and 'db_connection' in st.session_state:
        print("user_id before saving:", st.session_state.user_id)
        save_user_data(st.session_state.db_connection, st.session_state.user_id)

def add_income(date, source, amount, frequency="One-time", imported=False):
    new_income = pd.DataFrame({
        'date': [date],
        'source': [source],
        'amount': [amount],
        'frequency': [frequency],
        'imported': [imported]
    })
    # Duplicate prevention logic (unchanged)
    tx_str = f"{date}|{source}|{amount}|{frequency}".lower()
    tx_hash = hashlib.sha256(tx_str.encode()).hexdigest()
    if 'income_hashes' not in st.session_state:
        st.session_state.income_hashes = set()
    if tx_hash in st.session_state.income_hashes:
        st.warning("Duplicate income detected. This transaction was not added.")
        return
    st.session_state.income_hashes.add(tx_hash)
    st.session_state.income = pd.concat([st.session_state.income, new_income], ignore_index=True)
    # Save to database (if available) - use existing connection
    if st.session_state.get('mongodb_available', False) and 'db_connection' in st.session_state:
        print("user_id before saving:", st.session_state.user_id)
        save_user_data(st.session_state.db_connection, st.session_state.user_id)

def get_monthly_data(df, date_col='date'):
    if df.empty:
        return 0
    df[date_col] = pd.to_datetime(df[date_col])
    # Show all data, no date filtering
    return df['amount'].sum()

def generate_ai_insights():
    insights = []
    
    if not st.session_state.expenses.empty:
        expenses_df = st.session_state.expenses.copy()
        expenses_df['date'] = pd.to_datetime(expenses_df['date'])
        
        # Exclude 'INCOME' from spending categories
        spending_only = expenses_df[~expenses_df['category'].str.upper().eq('INCOME')]
        if not spending_only.empty:
            top_category = spending_only.groupby('category')['amount'].sum().idxmax()
            top_amount = spending_only.groupby('category')['amount'].sum().max()
            insights.append(f"💡 Your highest spending category is {top_category} with ₹{top_amount:,.2f}")
        
        # Weekly pattern analysis
        expenses_df['day_of_week'] = expenses_df['date'].dt.day_name()
        busiest_day = expenses_df.groupby('day_of_week')['amount'].sum().idxmax()
        insights.append(f"📅 You tend to spend the most on {busiest_day}s")
        
        # Budget recommendations
        monthly_expenses = get_monthly_data(st.session_state.expenses)
        monthly_income = get_monthly_data(st.session_state.income)
        
        if monthly_income > 0:
            savings_rate = ((monthly_income - monthly_expenses) / monthly_income) * 100
            if savings_rate < 20:
                insights.append("⚠️ Consider increasing your savings rate to at least 20% of income")
            else:
                insights.append(f"✅ Great job! Your savings rate is {savings_rate:.1f}%")
    
    return insights

# Main navigation
def main():
    # Initialize MongoDB
    db = init_mongodb()
    
    # Store database connection in session state for reuse
    if db is not None:
        st.session_state.db_connection = db
    
    # Show MongoDB status in sidebar
    if st.session_state.get('mongodb_available', False):
        st.sidebar.success("✅ MongoDB Connected")
    else:
        st.sidebar.warning("⚠️ MongoDB not available - using session storage")
        st.sidebar.info("Data will be stored in browser session only")
    
    # Authenticate user
    if not authenticate_user():
        st.info("Please login to access your financial dashboard.")
        return
    
    # Load user data
    if 'data_loaded' not in st.session_state:
        load_user_data(db, st.session_state.user_id)
        st.session_state.data_loaded = True
    
    st.markdown('<div class="main-header">💰 FinTrack - Personal Finance Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a section", 
                               ["Home", "Dashboard", "Add Transaction", "Budget", "Analytics", "Profile"])
    
    if page == "Home":
        home_page()
    elif page == "Dashboard":
        dashboard_page()
    elif page == "Add Transaction":
        add_transaction_page()
    elif page == "Budget":
        budget_page()
    elif page == "Analytics":
        analytics_page()
    elif page == "Profile":
        profile_page()

# Dashboard page (remains the same)
def dashboard_page():
    st.header("📊 Financial Overview")
    
    # --- Filter for manual/imported/all ---
    st.subheader("Transaction Type Filter")
    filter_type = st.selectbox(
        "Show Transactions:",
        ["All", "Manual only", "Imported only"],
        index=0,
        key="dashboard_filter_type"
    )
    # Filter expenses
    expenses_df = st.session_state.expenses.copy()
    if 'imported' in expenses_df.columns:
        if filter_type == "Manual only":
            expenses_df = expenses_df[(expenses_df['imported'] == False) | (expenses_df['imported'].isna())]
        elif filter_type == "Imported only":
            expenses_df = expenses_df[expenses_df['imported'] == True]
    elif filter_type != "All":
        st.info("No imported/manual distinction found in your data.")
    # Filter income
    income_df = st.session_state.income.copy()
    if 'imported' in income_df.columns:
        if filter_type == "Manual only":
            income_df = income_df[(income_df['imported'] == False) | (income_df['imported'].isna())]
        elif filter_type == "Imported only":
            income_df = income_df[income_df['imported'] == True]
    # Calculate key metrics using filtered data
    monthly_income = get_monthly_data(income_df)
    monthly_expenses = get_monthly_data(expenses_df)
    monthly_savings = monthly_income - monthly_expenses
    savings_rate = (monthly_savings / monthly_income * 100) if monthly_income > 0 else 0

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Monthly Income", f"₹{monthly_income:,.2f}")
    with col2:
        st.metric("Monthly Expenses", f"₹{monthly_expenses:,.2f}")
    with col3:
        st.metric("Monthly Savings", f"₹{monthly_savings:,.2f}")
    with col4:
        st.metric("Savings Rate", f"{savings_rate:.1f}%")

    # Charts row
    col1, col2 = st.columns(2)
    with col1:
        # Filter out 'INCOME' and any non-expense categories
        filtered_expenses = expenses_df[~expenses_df['category'].str.upper().eq('INCOME')]
        if not filtered_expenses.empty:
            st.subheader("💳 Spending by Category")
            expenses_by_category = filtered_expenses.groupby('category')['amount'].sum().reset_index()
            fig = px.pie(expenses_by_category, values='amount', names='category', 
                        title="Current Month Spending Distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No expense data available. Add some transactions to see the breakdown.")
    
    with col2:
        st.subheader("🎯 Savings Goals Progress")
        for goal in st.session_state.savings_goals:
            progress = min(goal['current'] / goal['target'], 1.0)
            st.write(f"**{goal['name']}**")
            st.progress(progress)
            st.write(f"₹{goal['current']:,} / ₹{goal['target']:,} ({progress*100:.1f}%)")
            st.write(f"Deadline: {goal['deadline']}")
            st.write("---")
    
    # AI Insights
    st.subheader("🤖 AI-Powered Insights")
    insights = generate_ai_insights()
    for insight in insights:
        st.info(insight)
    
    # Quick stats
    st.subheader("📈 Quick Stats")
    col1, col2, col3 = st.columns(3)
    with col1:
        if not expenses_df.empty:
            largest_expense = expenses_df['amount'].max()
            st.metric("Largest Expense", f"₹{largest_expense:,.2f}")
    with col2:
        if not expenses_df.empty:
            most_frequent = expenses_df['category'].mode().iloc[0] if not expenses_df.empty else "N/A"
            st.metric("Most Frequent Category", most_frequent)
    with col3:
        # Days until next salary (assuming monthly salary on 1st)
        today = datetime.now()
        if today.day == 1:
            days_until_salary = 0
        else:
            next_month = today.replace(day=1) + timedelta(days=32)
            next_salary = next_month.replace(day=1)
            days_until_salary = (next_salary - today).days
        st.metric("Days Until Next Salary", days_until_salary)

# Add transaction page (modified to include delete buttons)
def add_transaction_page():
    # --- DEBUG: Show last 10 expenses and their 'imported' status ---
    st.markdown('---')
    st.markdown('#### Debug: Last 10 Expenses')
    if not st.session_state.expenses.empty:
        debug_exp = st.session_state.expenses.tail(10)
        st.dataframe(debug_exp[['date', 'category', 'amount', 'description', 'imported']])
    else:
        st.info('No expenses recorded yet.')
    st.header("Add Transaction")

    # --- Danger Zone: Delete all CSV-imported transactions ---
    st.markdown("### Danger Zone")
    if st.button("🗑️ Delete ALL CSV-Imported Transactions"):
        # Only keep rows where imported is not True (i.e., keep manual entries)
        if 'imported' in st.session_state.expenses.columns:
            st.session_state.expenses = st.session_state.expenses[~st.session_state.expenses['imported']]
        if 'imported' in st.session_state.income.columns:
            st.session_state.income = st.session_state.income[~st.session_state.income['imported']]
        # Save to DB if available
        if st.session_state.get('mongodb_available', False) and 'db_connection' in st.session_state:
            save_user_data(st.session_state.db_connection, st.session_state.user_id)
        st.success("All CSV-imported transactions deleted! Manual entries are preserved.")
        st.rerun()

    # --- CSV/XLSX Upload Section ---
    st.subheader("📁 Upload Transactions (CSV/XLSX)")
    st.markdown("Upload your transactions file. Supports common Indian bank exports (SBI, ICICI, HDFC, Axis, etc.).")

    # Downloadable sample template
    sample_df = pd.DataFrame({
        'DATE': ['2024-01-01'],
        'AMOUNT': [-310.15],
        'SUBCATEGORY': ['Groceries'],
        'CATEGORY': ['Food'],
        'SOURCE': ['Checkings'],
        'DESCRIPTION': ['Groceries at supermarket XYZ']
    })
    sample_csv = sample_df.to_csv(index=False)
    st.download_button("Download Sample CSV", sample_csv, file_name="fintrack_sample.csv", mime="text/csv")

    uploaded_file = st.file_uploader("Upload CSV or XLSX file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        # Read file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.write("Preview:", df.head())

        # Common header mappings for Indian banks
        bank_header_map = [
            # SBI
            {'date': ['Txn Date', 'Date'], 'amount': ['Withdrawal Amt.', 'Deposit Amt.', 'Amount'], 'desc': ['Description', 'Narration', 'Particulars']},
            # ICICI
            {'date': ['Transaction Date'], 'amount': ['Amount'], 'desc': ['Remarks', 'Description']},
            # HDFC
            {'date': ['Date'], 'amount': ['Withdrawal Amount', 'Deposit Amount', 'Amount'], 'desc': ['Description', 'Narration']},
            # Axis
            {'date': ['Transaction Date'], 'amount': ['Amount'], 'desc': ['Description', 'Narration']},
        ]

        # Default mapping
        default_map = {
            'DATE': None,
            'AMOUNT': None,
            'CATEGORY': None,
            'SUBCATEGORY': None,
            'SOURCE': None,
            'DESCRIPTION': None
        }
        # Try to auto-map columns
        col_map = default_map.copy()
        for col in df.columns:
            col_lower = col.lower()
            if 'date' in col_lower:
                col_map['DATE'] = col
            elif 'amount' in col_lower:
                col_map['AMOUNT'] = col
            elif 'category' in col_lower:
                col_map['CATEGORY'] = col
            elif 'subcat' in col_lower:
                col_map['SUBCATEGORY'] = col
            elif 'source' in col_lower or 'account' in col_lower:
                col_map['SOURCE'] = col
            elif 'desc' in col_lower or 'narration' in col_lower or 'remark' in col_lower or 'particular' in col_lower:
                col_map['DESCRIPTION'] = col
        # UI for manual mapping
        st.markdown("#### Map Columns")
        for key in col_map:
            col_map[key] = st.selectbox(f"{key} column", [None] + list(df.columns), index=(1 + list(df.columns).index(col_map[key])) if col_map[key] in df.columns else 0, key=f"map_{key}")
        # Parse and add transactions
        if st.button("Import Transactions"):
            # Validate required columns
            required = ['DATE', 'AMOUNT', 'CATEGORY']
            missing = [k for k in required if not col_map[k]]
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            else:
                imported = 0
                skipped = 0
                duplicate_rows = []
                for idx, row in df.iterrows():
                    try:
                        date_val = pd.to_datetime(row[col_map['DATE']]).date()
                        amount_val = float(row[col_map['AMOUNT']])
                        category_val = str(row[col_map['CATEGORY']]) if col_map['CATEGORY'] else 'Miscellaneous'
                        subcat_val = row[col_map['SUBCATEGORY']] if col_map['SUBCATEGORY'] else ''
                        source_val = row[col_map['SOURCE']] if col_map['SOURCE'] else ''
                        desc_val = row[col_map['DESCRIPTION']] if col_map['DESCRIPTION'] else ''
                        # Duplicate check logic (same as add_expense/add_income)
                        if amount_val < 0:
                            tx_str = f"{date_val}|{source_val}|{abs(amount_val)}|One-time".lower()
                            tx_hash = hashlib.sha256(tx_str.encode()).hexdigest()
                            if 'income_hashes' not in st.session_state:
                                st.session_state.income_hashes = set()
                            if tx_hash in st.session_state.income_hashes:
                                skipped += 1
                                duplicate_rows.append(idx)
                                continue
                            add_income(date_val, source_val, abs(amount_val), "One-time", imported=True)
                        else:
                            tx_str = f"{date_val}|{category_val}|{amount_val}|{desc_val}".lower()
                            tx_hash = hashlib.sha256(tx_str.encode()).hexdigest()
                            if 'expense_hashes' not in st.session_state:
                                st.session_state.expense_hashes = set()
                            if tx_hash in st.session_state.expense_hashes:
                                skipped += 1
                                duplicate_rows.append(idx)
                                continue
                            add_expense(date_val, category_val, amount_val, desc_val, imported=True)
                        imported += 1
                    except Exception as e:
                        print(f"Row import error: {e}")
                        skipped += 1
                # --- CLEANUP: Ensure imported data is always correct type ---
                if not st.session_state.expenses.empty:
                    st.session_state.expenses['amount'] = pd.to_numeric(st.session_state.expenses['amount'], errors='coerce').fillna(0.0)
                    st.session_state.expenses['date'] = pd.to_datetime(st.session_state.expenses['date'], errors='coerce').dt.date
                    st.session_state.expenses['category'] = st.session_state.expenses['category'].astype(str).replace({None: 'Miscellaneous', '': 'Miscellaneous'}).fillna('Miscellaneous')
                if not st.session_state.income.empty:
                    st.session_state.income['amount'] = pd.to_numeric(st.session_state.income['amount'], errors='coerce').fillna(0.0)
                    st.session_state.income['date'] = pd.to_datetime(st.session_state.income['date'], errors='coerce').dt.date
                    st.session_state.income['source'] = st.session_state.income['source'].astype(str).replace({None: 'Other', '': 'Other'}).fillna('Other')
                # DEBUG: Show last 10 expenses and income after import
                st.write('DEBUG: Expenses after import:', st.session_state.expenses.tail(10))
                st.write('DEBUG: Income after import:', st.session_state.income.tail(10))
                # --- NEW: Show imported transactions only ---
                imported_exp = st.session_state.expenses[st.session_state.expenses['imported'] == True]
                st.markdown('---')
                st.markdown('#### Debug: Imported Expenses Only')
                if not imported_exp.empty:
                    st.dataframe(imported_exp[['date', 'category', 'amount', 'description', 'imported']].tail(10))
                else:
                    st.warning('No imported expenses found after import. If you expected imported transactions, check your CSV mapping and data.')
                st.info(f"Import summary: {imported} imported, {skipped} skipped (duplicates or errors).")
                if duplicate_rows:
                    st.caption(f"Rows skipped due to duplicates: {duplicate_rows}")
                st.success(f"Imported {imported} transactions!")
                st.rerun()
    # --- End Upload Section ---

    tab1, tab2 = st.tabs(["Add Expense", "Add Income"])
    
    with tab1:
        st.subheader("➖ Add Expense")
        with st.form("expense_form"):
            col1, col2 = st.columns(2)
            with col1:
                expense_date = st.date_input("Date", datetime.now())
                expense_category = st.selectbox("Category", EXPENSE_CATEGORIES)
            with col2:
                expense_amount = st.number_input("Amount (₹)", min_value=0.01, step=0.01)
                expense_description = st.text_input("Description (Optional)")
            
            if st.form_submit_button("Add Expense"):
                add_expense(expense_date, expense_category, expense_amount, expense_description)
                st.success(f"Added expense: ₹{expense_amount} for {expense_category}")
                st.rerun()
    
    with tab2:
        st.subheader("➕ Add Income")
        with st.form("income_form"):
            col1, col2 = st.columns(2)
            with col1:
                income_date = st.date_input("Date", datetime.now(), key="income_date")
                income_source = st.selectbox("Source", INCOME_SOURCES)
            with col2:
                income_amount = st.number_input("Amount (₹)", min_value=0.01, step=0.01, key="income_amount")
                income_frequency = st.selectbox("Frequency", ["One-time", "Monthly", "Weekly", "Yearly"])
            
            if st.form_submit_button("Add Income"):
                add_income(income_date, income_source, income_amount, income_frequency)
                st.success(f"Added income: ₹{income_amount} from {income_source}")
                st.rerun()
    
    # Recent transactions
    st.subheader("📝 Recent Transactions")
    
    if not st.session_state.expenses.empty or not st.session_state.income.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Recent Expenses**")
            if not st.session_state.expenses.empty:
                recent_expenses = st.session_state.expenses.tail(5).copy()
                if 'date' in recent_expenses.columns:
                    recent_expenses['date'] = recent_expenses['date'].astype(str)
                for idx, row in recent_expenses.iterrows():
                    st.write(row)
                    if st.button(f"Delete Expense {idx}", key=f"del_exp_{idx}"):
                        st.session_state.expenses.drop(idx, inplace=True)
                        st.session_state.expenses.reset_index(drop=True, inplace=True)
                        if st.session_state.get('mongodb_available', False) and 'db_connection' in st.session_state:
                            save_user_data(st.session_state.db_connection, st.session_state.user_id)
                        st.success("Expense deleted!")
                        st.rerun()
            else:
                st.info("No expenses recorded yet.")
        
        with col2:
            st.write("**Recent Income**")
            if not st.session_state.income.empty:
                recent_income = st.session_state.income.tail(5).copy()
                if 'date' in recent_income.columns:
                    recent_income['date'] = recent_income['date'].astype(str)
                for idx, row in recent_income.iterrows():
                    st.write(row)
                    if st.button(f"Delete Income {idx}", key=f"del_inc_{idx}"):
                        st.session_state.income.drop(idx, inplace=True)
                        st.session_state.income.reset_index(drop=True, inplace=True)
                        if st.session_state.get('mongodb_available', False) and 'db_connection' in st.session_state:
                            save_user_data(st.session_state.db_connection, st.session_state.user_id)
                        st.success("Income deleted!")
                        st.rerun()
            else:
                st.info("No income recorded yet.")

# Budget page (modified to save to DB)
def budget_page():
    st.header("💰 Budget Management")
    
    # Current budget overview
    total_budget = sum(st.session_state.budget.values())
    monthly_expenses = get_monthly_data(st.session_state.expenses)
    remaining_budget = total_budget - monthly_expenses
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Budget", f"₹{total_budget:,.2f}")
    with col2:
        st.metric("Total Spent", f"₹{monthly_expenses:,.2f}")
    with col3:
        color = "normal" if remaining_budget >= 0 else "inverse"
        st.metric("Remaining", f"₹{remaining_budget:,.2f}", delta=None)
    
    # Budget by category
    st.subheader("📊 Budget vs Actual Spending")
    
    if not st.session_state.expenses.empty:
        expenses_by_category = st.session_state.expenses.groupby('category')['amount'].sum().to_dict()
    else:
        expenses_by_category = {}
    
    budget_data = []
    for category, budget_amount in st.session_state.budget.items():
        spent = expenses_by_category.get(category, 0)
        remaining = budget_amount - spent
        budget_data.append({
            'Category': category,
            'Budget': budget_amount,
            'Spent': spent,
            'Remaining': remaining,
            'Usage %': (spent / budget_amount * 100) if budget_amount > 0 else 0
        })
    
    budget_df = pd.DataFrame(budget_data)
    
    # Display budget progress bars
    for _, row in budget_df.iterrows():
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.write(f"**{row['Category']}**")
            usage_pct = min(row['Usage %'] / 100, 1.0)
            st.progress(usage_pct)
            
            # Color coding based on usage
            if row['Usage %'] > 100:
                st.error(f"Over budget by ₹{abs(row['Remaining']):,.2f}")
            elif row['Usage %'] > 80:
                st.warning(f"₹{row['Remaining']:,.2f} remaining")
            else:
                st.success(f"₹{row['Remaining']:,.2f} remaining")
        
        with col2:
            st.metric("Budget", f"₹{row['Budget']:,.2f}")
        with col3:
            st.metric("Spent", f"₹{row['Spent']:,.2f}")
    
    # Edit budget section
    st.subheader("✏️ Edit Budget")
    with st.expander("Modify Budget Categories"):
        st.write("Adjust your budget for each category:")
        new_budget = {}
        
        col1, col2 = st.columns(2)
        categories = list(st.session_state.budget.keys())
        mid_point = len(categories) // 2
        
        with col1:
            for category in categories[:mid_point]:
                new_budget[category] = st.number_input(
                    f"{category}", 
                    value=st.session_state.budget[category],
                    min_value=0,
                    step=100,
                    key=f"budget_{category}"
                )
        
        with col2:
            for category in categories[mid_point:]:
                new_budget[category] = st.number_input(
                    f"{category}", 
                    value=st.session_state.budget[category],
                    min_value=0,
                    step=100,
                    key=f"budget_{category}"
                )
        
        if st.button("Update Budget"):
            st.session_state.budget = new_budget
            # Use stored database connection
            if st.session_state.get('mongodb_available', False) and 'db_connection' in st.session_state:
                save_user_data(st.session_state.db_connection, st.session_state.user_id)
            st.success("Budget updated successfully!")
            st.rerun()

# Analytics page (modified to save to DB)
def analytics_page():
    # ...existing code...
    # --- FILTERS ---
    st.subheader("🔎 Filters")
    expenses_df = st.session_state.expenses.copy()
    expenses_df['date'] = pd.to_datetime(expenses_df['date'])

    # Add filter for manual/imported/all
    filter_type = st.selectbox(
        "Show Transactions:",
        ["All", "Manual only", "Imported only"],
        index=0,
        key="analytics_filter_type"
    )
    if 'imported' in expenses_df.columns:
        if filter_type == "Manual only":
            expenses_df = expenses_df[(expenses_df['imported'] == False) | (expenses_df['imported'].isna())]
        elif filter_type == "Imported only":
            expenses_df = expenses_df[expenses_df['imported'] == True]
    elif filter_type != "All":
        st.info("No imported/manual distinction found in your data.")


    # --- DEBUG: Show filtered DataFrame for troubleshooting ---
    st.markdown('---')
    st.markdown('#### Debug: Filtered Expenses Data')
    st.write(f"Filter type: {filter_type}")
    st.dataframe(expenses_df[['date', 'category', 'amount', 'description', 'imported']] if 'imported' in expenses_df.columns else expenses_df)


    # Defensive: If no data, avoid errors in selectboxes/sliders
    if expenses_df.empty:
        st.warning("No expenses found. Please add transactions using the 'Add Transaction' page.")
        years = []
        categories = []
        min_amt, max_amt = 0.0, 0.0
    else:
        years = sorted([int(y) for y in expenses_df['date'].dt.year.dropna().unique()])
        categories = sorted([str(c) for c in expenses_df['category'].dropna().unique()])
        min_amt, max_amt = float(expenses_df['amount'].min()), float(expenses_df['amount'].max())

    # Only show 'None' if there is truly no year data
    if years:
        default_year_idx = len(years)-1
        year = st.selectbox("Year", years, index=default_year_idx)
        months = expenses_df[expenses_df['date'].dt.year == year]['date'].dt.month.dropna().unique()
        month_options = ["All"] + [datetime(1900, int(m), 1).strftime('%B') for m in sorted(months)]
    else:
        year = None
        month_options = ["All"]
    col1, col2, col3 = st.columns(3)
    with col1:
        year_val = year if year is not None else "None"
        month = st.selectbox("Month", month_options)
    with col2:
        selected_cats = st.multiselect("Category", categories, default=categories)
    with col3:
        if expenses_df.empty or np.isnan(min_amt) or np.isnan(max_amt):
            amt_range = (0.0, 0.0)
            st.info("No amount data available.")
        elif min_amt == max_amt:
            amt_range = (min_amt, max_amt)
            st.info(f"Only one unique amount: {min_amt}. No range to filter.")
        else:
            amt_range = st.slider("Amount Range", min_amt, max_amt, (min_amt, max_amt))


    # Apply filters
    if expenses_df.empty or year is None:
        filtered = expenses_df.iloc[0:0]
    else:
        filtered = expenses_df[expenses_df['date'].dt.year == year]
        if month != "All" and years:
            try:
                month_num = datetime.strptime(month, '%B').month
                filtered = filtered[filtered['date'].dt.month == month_num]
            except Exception:
                pass
        if selected_cats:
            filtered = filtered[filtered['category'].isin(selected_cats)]
        if amt_range:
            filtered = filtered[(filtered['amount'] >= amt_range[0]) & (filtered['amount'] <= amt_range[1])]
    

    # --- Show warning if no data for selected year/filter ---
    if filtered.empty:
        if expenses_df.empty:
            st.info("No expenses found. Please add transactions using the 'Add Transaction' page.")
        elif filter_type == "Manual only":
            st.warning(f"No manually added transactions found for {year_val} with the selected filters.")
        elif filter_type == "Imported only":
            st.warning(f"No imported transactions found for {year_val} with the selected filters.")
        else:
            st.warning(f"No transactions found for {year_val} with the selected filters.")
    
    # --- Spending Trends ---
    st.subheader("💸 Spending Trends")
    if not filtered.empty:
        # Daily spending trend with trendline
        daily_spending = filtered.groupby('date')['amount'].sum().reset_index()
        fig = px.line(daily_spending, x='date', y='amount', title="Daily Spending Trend", markers=True)
        # Add trendline
        if len(daily_spending) > 1:
            import statsmodels.api as sm
            x = (daily_spending['date'] - daily_spending['date'].min()).dt.days.values.reshape(-1, 1)
            y = daily_spending['amount'].values
            model = sm.OLS(y, sm.add_constant(x)).fit()
            trend = model.predict(sm.add_constant(x))
            fig.add_traces(go.Scatter(x=daily_spending['date'], y=trend, mode='lines', name='Trendline'))
        st.plotly_chart(fig, use_container_width=True)
        
        # Category analysis (horizontal bar)
        st.subheader("📊 Top Spending Categories")
        category_spending = filtered.groupby('category')['amount'].sum().sort_values(ascending=True)
        fig = px.bar(
            x=category_spending.values,
            y=category_spending.index,
            orientation='h',
            title="Spending by Category",
            labels={'x': 'Amount (₹)', 'y': 'Category'},
            hover_data={'Amount (₹)': category_spending.values, 'Category': category_spending.index}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Weekly pattern
        st.subheader("📅 Weekly Spending Pattern")
        filtered['day_of_week'] = filtered['date'].dt.day_name()
        weekly_pattern = filtered.groupby('day_of_week')['amount'].sum()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_pattern = weekly_pattern.reindex(day_order, fill_value=0)
        fig = px.bar(x=weekly_pattern.index, y=weekly_pattern.values, title="Spending by Day of Week")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No expenses match the selected filters.")
    
    # --- Income vs Expenses ---
    if not st.session_state.income.empty and not filtered.empty:
        st.subheader("💰 Income vs Expenses")
        income_df = st.session_state.income.copy()
        income_df['date'] = pd.to_datetime(income_df['date'])
        # Filter income by year/month
        income_filtered = income_df[income_df['date'].dt.year == year]
        if month != "All":
            income_filtered = income_filtered[income_filtered['date'].dt.month == month_num]
        # Monthly comparison
        expenses_monthly = filtered.groupby(filtered['date'].dt.to_period('M'))['amount'].sum()
        income_monthly = income_filtered.groupby(income_filtered['date'].dt.to_period('M'))['amount'].sum()
        comparison_df = pd.DataFrame({
            'Month': expenses_monthly.index.astype(str),
            'Expenses': expenses_monthly.values,
            'Income': income_monthly.reindex(expenses_monthly.index, fill_value=0).values
        })
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Income', x=comparison_df['Month'], y=comparison_df['Income']))
        fig.add_trace(go.Bar(name='Expenses', x=comparison_df['Month'], y=comparison_df['Expenses']))
        fig.update_layout(title="Monthly Income vs Expenses", barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    # --- Goal vs Actual (Progress Bars) ---
    st.subheader("🎯 Goal vs Actual Savings Progress")
    if 'savings_goals' in st.session_state:
        for goal in st.session_state.savings_goals:
            progress = min(goal['current'] / goal['target'], 1.0)
            st.write(f"**{goal['name']}**")
            st.progress(progress)
            st.write(f"₹{goal['current']:,} / ₹{goal['target']:,} ({progress*100:.1f}%)")
            st.write(f"Deadline: {goal['deadline']}")
            st.write("---")
    
    # --- Financial health metrics ---
    st.subheader("🏥 Financial Health Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_daily_spend = filtered['amount'].sum() / max(len(filtered['date'].unique()), 1) if not filtered.empty else 0
        st.metric("Average Daily Spend", f"₹{avg_daily_spend:.2f}")
    with col2:
        monthly_income = get_monthly_data(st.session_state.income)
        monthly_expenses = get_monthly_data(st.session_state.expenses)
        savings_rate = ((monthly_income - monthly_expenses) / monthly_income * 100) if monthly_income > 0 else 0
        st.metric("Current Savings Rate", f"{savings_rate:.1f}%")
    with col3:
        if not filtered.empty:
            largest_expense = filtered['amount'].max()
            st.metric("Largest Single Expense", f"₹{largest_expense:.2f}")

# Profile page (modified to save to DB)
def profile_page():
    st.header("👤 User Profile")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Generate avatar with first letter of user's name
        user_name = st.session_state.user_profile.get('name', 'U')
        first_letter = user_name[0].upper() if user_name else 'U'
        avatar_svg = f'''
        <svg width="120" height="120" xmlns="http://www.w3.org/2000/svg">
          <circle cx="60" cy="60" r="58" fill="#1f77b4" />
          <text x="50%" y="58%" text-anchor="middle" fill="#fff" font-size="60" font-family="Arial, sans-serif" dy=".3em">{first_letter}</text>
        </svg>
        '''
        st.markdown(f'<div style="text-align:center; max-width:130px; height:130px; margin:auto; overflow:hidden; display:flex; align-items:center; justify-content:center;">{avatar_svg}</div>', unsafe_allow_html=True)
        st.caption("Profile Picture")
    
    with col2:
        st.subheader("Profile Information")
        
        with st.form("profile_form"):
            name = st.text_input("Name", value=st.session_state.user_profile['name'])
            email = st.text_input("Email", value=st.session_state.user_profile['email'], disabled=True)
            member_since = st.text_input("Member Since", value=st.session_state.user_profile['member_since'], disabled=True)
            
            if st.form_submit_button("Update Profile"):
                st.session_state.user_profile['name'] = name
                # Use stored database connection
                if st.session_state.get('mongodb_available', False) and 'db_connection' in st.session_state:
                    save_user_data(st.session_state.db_connection, st.session_state.user_id)
                st.success("Profile updated successfully!")
    
    # Account statistics
    st.subheader("📊 Account Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_transactions = len(st.session_state.expenses) + len(st.session_state.income)
        st.metric("Total Transactions", total_transactions)
    
    with col2:
        total_expenses = st.session_state.expenses['amount'].sum() if not st.session_state.expenses.empty else 0
        st.metric("Total Expenses", f"₹{total_expenses:,.2f}")
    
    with col3:
        total_income = st.session_state.income['amount'].sum() if not st.session_state.income.empty else 0
        st.metric("Total Income", f"₹{total_income:,.2f}")
    
    with col4:
        net_worth = total_income - total_expenses
        st.metric("Net Position", f"₹{net_worth:,.2f}")

def home_page():
    st.markdown("""
    <div class="main-header">Take Control of Your Finances</div>
    <p>Welcome to a Streamlit-based personal finance dashboard. This intuitive dashboard is designed to give you a visual representation of your finances over time, empowering you to make informed decisions and achieve your financial goals.</p>
    <h3>What can you see here?</h3>
    <ul>
        <li><b>Track your income and expenses</b> 📊: See exactly where your money comes from and goes. Easy-to-read visualizations break down your income streams and spending habits, helping you identify areas for potential savings or growth. Gain a comprehensive understanding of your financial patterns to make informed decisions about budgeting and resource allocation.</li>
        <li><b>Monitor your cash flow</b> 🐝: Stay on top of your incoming and outgoing funds. This dashboard provides clear insight into your current financial liquidity, allowing you to plan for upcoming expenses and avoid potential shortfalls. Anticipate cash crunches and optimize your spending timing to maintain a healthy financial balance.</li>
        <li><b>View your financial progress</b> 📈: Charts and graphs track your progress towards your financial goals over time. Whether you're saving for a dream vacation or planning for retirement, this dashboard keeps you motivated and on track. Visualize your long-term financial journey and adjust your strategies based on real-time performance data.</li>
    </ul>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.header("FAQ")
    with st.expander("How do I add my transactions?"):
        st.write("You can add transactions manually or upload a CSV/XLSX file exported from your bank. Use the 'Add Transaction' page and follow the instructions for mapping columns if uploading a file.")
    with st.expander("Can I edit or delete a transaction after adding it?"):
        st.write("Yes! You can delete recent transactions directly from the 'Add Transaction' page. Editing is currently not supported, but you can delete and re-add a corrected entry.")
    with st.expander("How do I set or change my budget?"):
        st.write("Go to the 'Budget' section to view, set, or update your monthly budget for each category. Changes are saved automatically if you are connected to MongoDB.")
    with st.expander("What happens if I lose connection to MongoDB?"):
        st.write("If the app can't connect to MongoDB, your data will only be saved for the current browser session. You'll see a warning at the top of the app if this happens.")
    with st.expander("How does the app prevent duplicate transactions?"):
        st.write("When you add or upload transactions, the app checks for duplicates using the date, category/source, amount, and description/frequency. Duplicate entries are not added.")
    with st.expander("Is my data private and secure?"):
        st.write("Your data is stored securely in your own MongoDB database. Only you can access your data using your login credentials. Passwords are hashed for security.")

if __name__ == "__main__":
    main()