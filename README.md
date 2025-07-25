# FinTrack - Personal Finance Dashboard

A modern, secure, and user-friendly personal finance dashboard built with Streamlit. Track your expenses, income, budgets, and savings goals with powerful analytics and a beautiful, mobile-friendly UI.

---

## 🚀 Features

- 🔐 **Secure Login**: Email + password authentication with hashed passwords
- 🏠 **Home Page & FAQ**: Friendly landing page with onboarding and helpful answers
- 💰 **Expense & Income Tracking**: Add, categorize, and manage your financial transactions
- 📁 **CSV/XLSX Upload**: Upload bank statements, map columns, and prevent duplicates automatically
- 📊 **Dashboard**: Visual overview of your financial health
- 💳 **Budget Management**: Set and track budgets by category, with progress bars
- 📈 **Analytics**: Filter by time, category, and amount; trendlines and interactive charts
- 🎯 **Savings Goals**: Track progress towards your goals with visual feedback
- 🤖 **AI Insights**: Smart recommendations based on your spending patterns
- 📱 **Mobile Responsive**: Clean, modern UI that works on any device
- 🛡️ **Data Privacy**: Your data is stored securely in your own MongoDB database
- 🛑 **Duplicate Prevention**: No more double entries—duplicates are detected and blocked
- ⚠️ **Error Handling**: Clear feedback if MongoDB is unavailable (session-only mode)

---

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd fintrack---personal-finance-dashboard
```
2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
streamlit run connect.py
```

---

## 🗄️ MongoDB Setup (Optional but recommended)

- The app works with or without MongoDB.
- **With MongoDB:** Data is saved permanently.
- **Without MongoDB:** Data is stored only for your browser session.

**To use MongoDB Atlas or your own server:**
1. Set your connection string in `config.py` or as environment variables:
    ```bash
    export MONGODB_URI="mongodb://your-mongodb-host:27017/"
    export MONGODB_DB_NAME="your_database_name"
    ```
2. Make sure your database is running and accessible.

---

## 📁 File Upload Format & Mapping

You can upload transactions from your bank statement (CSV/XLSX). The app supports common Indian bank formats (SBI, ICICI, HDFC, Axis, etc.) and lets you map columns easily.

**Required columns for upload:**
| Column      | Type   | Required | Example                |
|-------------|--------|----------|------------------------|
| DATE        | string | Yes      | 2024-01-01             |
| AMOUNT      | float  | Yes      | -310.15 (expense)      |
| CATEGORY    | string | Yes      | Food                   |
| SUBCATEGORY | string | No       | Groceries              |
| SOURCE      | string | No       | Checkings              |
| DESCRIPTION | string | No       | Groceries at XYZ store |

- **Upload your bank file** on the "Add Transaction" page.
- **Map columns** using the UI (auto-mapping for common banks).
- **Duplicates are automatically detected and blocked.**
- Download a sample CSV from the app for reference.

---

## 💡 Usage

1. **Login/Register:** Use your email and password to access your dashboard.
2. **Home:** See a welcome message and FAQ.
3. **Add Transactions:** Manually add or upload your expenses/income. Map columns as needed.
4. **Dashboard:** Get a quick overview of your finances.
5. **Budget:** Set and track your monthly budget by category.
6. **Analytics:** Filter, visualize, and analyze your spending and income trends.
7. **Profile:** View your stats and update your profile info.

---

## 📱 Mobile & UX
- The app is fully responsive and works great on mobile devices.
- Tables and charts are scrollable and adapt to small screens.
- Modern theme and icons for a delightful experience.

---

## 🛡️ Data Storage & Security
- **With MongoDB:** Data is stored securely and permanently.
- **Without MongoDB:** Data is stored in your browser session (temporary).
- **Passwords are hashed** and never stored in plain text.
- **Clear error messages** if the database is unavailable.

---

## 🧩 Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License
This project is licensed under the MIT License. 