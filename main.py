from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import re
import os
from datetime import datetime, timedelta

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'default_secret_key')
app.permanent_session_lifetime = timedelta(minutes=30)

# Constants
MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
DATASET_PATH = 'Dataset.xlsx'

# Global expense columns (will be initialized properly)
EXPENSE_COLUMNS = []

# Initialize expense columns from dataset or use defaults
def get_expense_columns():
    global EXPENSE_COLUMNS
    try:
        df = pd.read_excel(DATASET_PATH)
        columns = [col for col in df.columns if col not in ['Month', 'Year', 'Total_Income', 'Total_Usage', 'Savings', 'Savings_Percentage'] and pd.api.types.is_numeric_dtype(df[col])]
        EXPENSE_COLUMNS = columns
        return columns
    except Exception as e:
        print(f"Error loading expense columns: {str(e)}")
        default_columns = ['Rent', 'Utilities', 'Groceries', 'Transportation', 'Entertainment', 'Healthcare', 'Misc']
        EXPENSE_COLUMNS = default_columns
        return default_columns

# Initialize the columns when the module loads
get_expense_columns()

# Helper functions
def load_data():
    global EXPENSE_COLUMNS
    try:
        df = pd.read_excel(DATASET_PATH)
        if df.empty:
            raise ValueError("Empty dataset")
        # Ensure proper data structure
        if 'Month' not in df.columns:
            df['Month'] = MONTHS[:len(df)] if len(df) <= 12 else [f"Month {i+1}" for i in range(len(df))]
        # Add Year column if it doesn't exist
        if 'Year' not in df.columns:
            df['Year'] = datetime.now().year
        # Ensure numeric columns are properly typed
        numeric_columns = ['Total_Income', 'Total_Usage', 'Savings', 'Savings_Percentage', 'Year'] + EXPENSE_COLUMNS
        for col in numeric_columns:
            if col in df.columns:
                if col == 'Year':
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(datetime.now().year).astype(int)
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        if 'Total_Usage' not in df.columns:
            if EXPENSE_COLUMNS:
                df['Total_Usage'] = df[EXPENSE_COLUMNS].sum(axis=1)
            else:
                df['Total_Usage'] = 0
        if 'Savings_Percentage' not in df.columns and 'Total_Income' in df.columns:
            df['Savings'] = df['Total_Income'] - df['Total_Usage']
            df['Savings_Percentage'] = (df['Savings'] / df['Total_Income']) * 100
            df['Savings_Percentage'] = df['Savings_Percentage'].fillna(0)
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return pd.DataFrame(columns=['Month', 'Year'] + EXPENSE_COLUMNS + ['Total_Income', 'Total_Usage', 'Savings', 'Savings_Percentage'])

def save_data(df):
    try:
        df.to_excel(DATASET_PATH, index=False)
        return True
    except Exception as e:
        print(f"Error saving data: {str(e)}")
        return False

def get_next_month_and_year(df):
    """Get the next month and year based on the last entry in the dataset"""
    if df.empty:
        current_date = datetime.now()
        return MONTHS[current_date.month - 1], current_date.year
    try:
        # Get the last row's month and year
        last_row = df.iloc[-1]
        if 'Month' in df.columns and 'Year' in df.columns:
            last_month = str(last_row['Month'])
            last_year = int(last_row['Year']) if pd.notna(last_row['Year']) else datetime.now().year
            # Find the index of the last month
            try:
                month_index = MONTHS.index(last_month)
                if month_index == 11:  # December
                    return MONTHS[0], last_year + 1  # January of next year
                else:
                    return MONTHS[month_index + 1], last_year
            except ValueError:
                # If month name is not in standard format, use current date
                current_date = datetime.now()
                return MONTHS[current_date.month - 1], current_date.year
        else:
            # If no proper month/year columns, use current date
            current_date = datetime.now()
            return MONTHS[current_date.month - 1], current_date.year
    except Exception as e:
        print(f"Error determining next month: {str(e)}")
        current_date = datetime.now()
        return MONTHS[current_date.month - 1], current_date.year

def predict_next_month(df):
    if len(df) < 3:
        print("Need at least 3 months of data for prediction")
        return None
    try:
        df = df.copy()
        df['Previous_Usage'] = df['Total_Usage'].shift(1)
        df['Two_Months_Ago'] = df['Total_Usage'].shift(2)
        df.dropna(inplace=True)
        if len(df) < 1:
            return None
        X = df[['Previous_Usage', 'Two_Months_Ago']]
        y = df['Total_Usage']
        model = LinearRegression()
        model.fit(X, y)
        last_two = df[['Previous_Usage', 'Two_Months_Ago']].iloc[-1].values.reshape(1, -1)
        return model.predict(last_two)[0]
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return None

def predict_category_breakdown(df, total_prediction):
    global EXPENSE_COLUMNS
    if len(df) < 1:
        return None
    last_month = df.iloc[-1][EXPENSE_COLUMNS]
    proportions = last_month / last_month.sum()
    return (proportions * total_prediction).to_dict()

def calculate_savings_percentage(income, expenses):
    total_spent = sum(expenses.values())
    savings = income - total_spent
    return (savings / income) * 100 if income > 0 else 0

def validate_amount(amount_str):
    try:
        amount = float(str(amount_str).strip())
        return amount >= 0
    except (ValueError, TypeError):
        return False

# Session Type Constants
SESSION_TYPE_PREDICTION = 'prediction'
SESSION_TYPE_ADDITION = 'addition'
SESSION_TYPE_SUGGESTION = 'suggestion'
SESSION_TYPE_CATEGORY = 'category'
SESSION_TYPE_EXPECTATION = 'expectation'

# Session management functions
def start_session(session_type):
    session.clear()
    session['session_type'] = session_type
    session['state'] = 'start'
    session['data'] = {}

def end_session():
    session.clear()

def is_valid_session():
    return 'session_type' in session

def get_session_specific_error_message(session_type):
    """Get appropriate error message based on current session"""
    if session_type == SESSION_TYPE_PREDICTION:
        return "❌ Invalid input for prediction session. Please answer with 'yes' or 'no', or say 'menu' to return to main menu."
    elif session_type == SESSION_TYPE_ADDITION:
        return "❌ Invalid input for expense addition. Please provide the requested information, or say 'menu' to return to main menu."
    elif session_type == SESSION_TYPE_SUGGESTION:
        return "❌ Invalid format for suggestion. Please specify amount and category like: 'Can I spend $500 on Groceries?', or say 'menu' to return to main menu."
    elif session_type == SESSION_TYPE_CATEGORY:
        return "❌ Invalid input for category management. Please follow the instructions or say 'menu' to return to main menu."
    elif session_type == SESSION_TYPE_EXPECTATION:
        return "❌ Invalid input for expectation session. Please provide the requested information, or say 'menu' to return to main menu."
    else:
        return "❌ Invalid input. Say 'menu' to see available options or 'bye' to exit."

@app.route('/')
def index():
    end_session()
    return render_template('index.html')

@app.route('/get_chart_data')
def get_chart_data():
    try:
        df = load_data()
        if df.empty:
            return jsonify({
                "error": "No data available",
                "months": [],
                "usage": [],
                "income": [],
                "savings": []
            })
        # Ensure we have the required columns
        if 'Month' not in df.columns:
            df['Month'] = MONTHS[:len(df)] if len(df) <= 12 else [f"Month {i+1}" for i in range(len(df))]
        if 'Total_Usage' not in df.columns:
            if EXPENSE_COLUMNS:
                df['Total_Usage'] = df[EXPENSE_COLUMNS].sum(axis=1)
            else:
                df['Total_Usage'] = 0
        # Handle sorting safely
        try:
            # Only sort by categorical if months match our expected format
            valid_months = [month for month in df['Month'] if month in MONTHS]
            if len(valid_months) == len(df):
                df['Month'] = pd.Categorical(df['Month'], categories=MONTHS, ordered=True)
                df = df.sort_values('Month')
        except Exception as sort_error:
            print(f"Sorting warning: {sort_error}")
            # If sorting fails, keep original order
            pass
        # Prepare chart data with safe defaults
        months_display = []
        if 'Year' in df.columns:
            # Combine month and year for display
            for i, row in df.iterrows():
                month = str(row['Month'])
                year = str(int(row['Year'])) if pd.notna(row['Year']) else ''
                months_display.append(f"{month} {year}" if year else month)
        else:
            months_display = df['Month'].astype(str).tolist()
        chart_data = {
            'months': months_display,
            'usage': df['Total_Usage'].fillna(0).tolist(),  # Fill NaN with 0
            'income': df['Total_Income'].fillna(0).tolist() if 'Total_Income' in df.columns else [0] * len(df),
            'savings': df['Savings_Percentage'].fillna(0).tolist() if 'Savings_Percentage' in df.columns else [0] * len(df)
        }
        return jsonify(chart_data)
    except Exception as e:
        print(f"Chart data error: {str(e)}")
        return jsonify({
            "error": f"Failed to load chart data: {str(e)}",
            "months": [],
            "usage": [],
            "income": [],
            "savings": []
        })

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '').strip().lower()
    df = load_data()

    # Handle greetings
    if any(greeting in user_input for greeting in ['hi', 'hello', 'hey']):
        end_session()
        return jsonify({
            "response": "Welcome to your Personal Finance Assistant! 🎯\n"
                        "I can help you with:\n"
                        "1. Expense predictions\n"
                        "2. Adding new expenses\n"
                        "3. Financial suggestions\n"
                        "4. Managing categories\n"
                        "5. Setting expectations\n"
                        "What would you like to do? (Say 'predict', 'add', 'suggest', 'category', or 'expect')"
        })
    
    # Handle exit commands
    if any(word in user_input for word in ['bye', 'exit', 'quit']):
        end_session()
        return jsonify({"response": "Goodbye! Come back anytime for financial advice. 💰"})
    
    # Handle menu/help commands
    if 'menu' in user_input or 'help' in user_input:
        end_session()
        return jsonify({
            "response": "Main Menu 📋\n"
                        "1. Predict expenses - Say 'predict'\n"
                        "2. Add new expenses - Say 'add'\n"
                        "3. Get suggestions - Say 'suggest'\n"
                        "4. Manage categories - Say 'category'\n"
                        "5. Set expectations - Say 'expect'\n"
                        "6. Exit - Say 'bye'"
        })

    # If user is already in a session, handle it within that session context
    if is_valid_session():
        current_session_type = session['session_type']
        
        if current_session_type == SESSION_TYPE_PREDICTION:
            return handle_prediction_session(df, user_input)
        elif current_session_type == SESSION_TYPE_ADDITION:
            return handle_addition_session(df, user_input)
        elif current_session_type == SESSION_TYPE_SUGGESTION:
            return handle_suggestion_session(df, user_input)
        elif current_session_type == SESSION_TYPE_CATEGORY:
            return handle_category_session(df, user_input)
        elif current_session_type == SESSION_TYPE_EXPECTATION:
            return handle_expectation_session(df, user_input)
    
    # If no active session, check for session initialization keywords
    if 'predict' in user_input or 'forecast' in user_input:
        return handle_prediction_session(df, user_input)
    elif ('add' in user_input or 'update' in user_input) and 'expense' in user_input:
        return handle_addition_session(df, user_input)
    elif 'add' in user_input and 'new' in user_input:
        return handle_addition_session(df, user_input)
    elif user_input.strip() == 'add':  # Only 'add' by itself
        return handle_addition_session(df, user_input)
    elif 'suggest' in user_input or 'advice' in user_input:
        return handle_suggestion_session(df, user_input)
    elif 'category' in user_input or 'categories' in user_input:
        return handle_category_session(df, user_input)
    elif 'expect' in user_input or 'goal' in user_input:
        return handle_expectation_session(df, user_input)

    # Default response for unrecognized input
    return jsonify({
        "response": "❌ I didn't understand that. Available commands:\n"
                   "• 'predict' - Get expense predictions\n"
                   "• 'add' - Add new monthly expenses\n"
                   "• 'suggest' - Get spending suggestions\n"
                   "• 'category' - Manage expense categories\n"
                   "• 'expect' - Set financial expectations\n"
                   "• 'menu' - Show main menu\n"
                   "• 'bye' - Exit\n\nWhat would you like to do?"
    })


# Session handlers
def handle_prediction_session(df, user_input):
    if not is_valid_session() or session['session_type'] != SESSION_TYPE_PREDICTION:
        start_session(SESSION_TYPE_PREDICTION)
        session['state'] = 'predict_total'
    
    try:
        if session['state'] == 'predict_total':
            prediction = predict_next_month(df)
            if prediction is None:
                end_session()
                return jsonify({"response": "❌ I need at least 3 months of data to make predictions. Please add more expense data first."})
            session['total_prediction'] = prediction
            session['state'] = 'confirm_breakdown'
            return jsonify({
                "response": f"📊 Next month's predicted total expenses: ${prediction:.2f}\n"
                           "Would you like to see the category breakdown? (yes/no)"
            })
        elif session['state'] == 'confirm_breakdown':
            if 'yes' in user_input:
                breakdown = predict_category_breakdown(df, session['total_prediction'])
                if breakdown is None:
                    end_session()
                    return jsonify({"response": "❌ Couldn't calculate category breakdown due to insufficient data."})
                response = ["💰 Predicted Category Breakdown:"]
                for cat, amt in breakdown.items():
                    response.append(f"{cat}: ${amt:.2f}")
                end_session()
                return jsonify({
                    "response": "\n".join(response) + "\n\nSay 'menu' for more options or 'bye' to exit."
                })
            elif 'no' in user_input:
                end_session()
                return jsonify({
                    "response": "Okay! Say 'predict' anytime for new predictions or 'menu' for options."
                })
            else:
                return jsonify({
                    "response": get_session_specific_error_message(SESSION_TYPE_PREDICTION)
                })
    except Exception as e:
        print(f"Prediction session error: {str(e)}")
        end_session()
        return jsonify({"response": "❌ An error occurred during prediction. Please try again."})
    
    end_session()
    return jsonify({"response": "Prediction session ended. Say 'menu' for options."})


def handle_addition_session(df, user_input):
    global EXPENSE_COLUMNS
    if not is_valid_session() or session['session_type'] != SESSION_TYPE_ADDITION:
        start_session(SESSION_TYPE_ADDITION)
        session['state'] = 'confirm_income_update'
        
        # Determine next month and year
        next_month, next_year = get_next_month_and_year(df)
        session['new_data'] = {'Month': next_month, 'Year': next_year}
        
        return jsonify({
            "response": f"🆕 New Expense Entry Session for {next_month} {next_year}\nWould you like to update your income for this month? (yes/no)"
        })

    try:
        if session['state'] == 'confirm_income_update':
            if 'yes' in user_input:
                session['state'] = 'get_income'
                return jsonify({"response": "Please enter your new total income for this month:"})
            elif 'no' in user_input:
                if len(df) > 0 and 'Total_Income' in df.columns:
                    # Convert pandas types to Python native types for JSON serialization
                    last_income = df.iloc[-1]['Total_Income']
                    session['new_data']['Total_Income'] = float(last_income) if pd.notna(last_income) else 0.0
                else:
                    session['new_data']['Total_Income'] = 0.0
                session['state'] = 'get_expenses'
                session['current_category'] = 0
                return jsonify({
                    "response": f"Using previous income: ${session['new_data']['Total_Income']:.2f}\nPlease enter amount for {EXPENSE_COLUMNS[0]}:"
                })
            else:
                return jsonify({"response": get_session_specific_error_message(SESSION_TYPE_ADDITION)})

        elif session['state'] == 'get_income':
            if validate_amount(user_input):
                # Ensure we store as Python float, not pandas type
                session['new_data']['Total_Income'] = float(user_input)
                session['state'] = 'get_expenses'
                session['current_category'] = 0
                return jsonify({
                    "response": f"Great! Income set to ${float(user_input):.2f}\nNow please enter amount for {EXPENSE_COLUMNS[0]}:"
                })
            else:
                return jsonify({
                    "response": "❌ Invalid amount. Please enter a positive number for your income:"
                })

        elif session['state'] == 'get_expenses':
            cat_index = session['current_category']
            current_cat = EXPENSE_COLUMNS[cat_index]
            if validate_amount(user_input):
                # Ensure we store as Python float, not pandas type
                session['new_data'][current_cat] = float(user_input)
                if cat_index + 1 < len(EXPENSE_COLUMNS):
                    session['current_category'] += 1
                    next_cat = EXPENSE_COLUMNS[cat_index + 1]
                    return jsonify({
                        "response": f"✅ {current_cat}: ${float(user_input):.2f}\nNow enter amount for {next_cat}:"
                    })
                else:
                    # Calculate totals using Python native types
                    total_usage = sum([float(session['new_data'][cat]) for cat in EXPENSE_COLUMNS if cat in session['new_data']])
                    session['new_data']['Total_Usage'] = float(total_usage)
                    
                    if 'Total_Income' in session['new_data']:
                        income = float(session['new_data']['Total_Income'])
                        savings = income - total_usage
                        session['new_data']['Savings'] = float(savings)
                        session['new_data']['Savings_Percentage'] = float((savings / income) * 100) if income > 0 else 0.0
                    
                    # Create new row and save to dataset
                    try:
                        # Convert all values to appropriate types for pandas
                        new_row_data = {}
                        for key, value in session['new_data'].items():
                            if key == 'Month':
                                new_row_data[key] = str(value)
                            elif key == 'Year':
                                new_row_data[key] = int(value)
                            else:
                                new_row_data[key] = float(value) if value is not None else 0.0
                        
                        # Ensure Year column exists in original dataframe
                        if 'Year' not in df.columns:
                            df['Year'] = datetime.now().year
                        
                        new_df = pd.concat([df, pd.DataFrame([new_row_data])], ignore_index=True)
                        
                        # Get savings percentage safely
                        savings_percentage = session['new_data'].get('Savings_Percentage', 0.0)
                        month_year = f"{session['new_data']['Month']} {session['new_data']['Year']}"
                        
                        if save_data(new_df):
                            end_session()
                            return jsonify({
                                "response": "✅ Successfully added new expenses!\n"
                                           f"Month: {month_year}\n"
                                           f"Total Expenses: ${total_usage:.2f}\n"
                                           f"Savings: {savings_percentage:.1f}%\n"
                                           "\nSay 'menu' for options or 'predict' to see forecasts."
                            })
                        else:
                            end_session()
                            return jsonify({
                                "response": "❌ Failed to save data. Please try again later."
                            })
                            
                    except Exception as e:
                        print(f"Error saving new data: {str(e)}")
                        end_session()
                        return jsonify({
                            "response": "❌ Failed to save data due to a system error. Please try again."
                        })
            else:
                return jsonify({
                    "response": f"❌ Invalid amount for {current_cat}. Please enter a positive number:"
                })

    except Exception as e:
        print(f"Addition session error: {str(e)}")
        end_session()
        return jsonify({"response": "❌ An error occurred during expense addition. Please try again."})

    end_session()
    return jsonify({"response": "Addition session ended. Say 'menu' for options."})

def handle_suggestion_session(df, user_input):
    global EXPENSE_COLUMNS
    if not is_valid_session() or session['session_type'] != SESSION_TYPE_SUGGESTION:
        start_session(SESSION_TYPE_SUGGESTION)
        return jsonify({
            "response": "💡 Suggestion Session Started\n"
                       "Ask me about spending plans like:\n"
                       "• 'Can I spend $1000 on Rent?'\n"
                       "• 'Is $200 for Entertainment okay?'\n"
                       "• 'Should I budget $300 for Groceries?'\n\n"
                       "What would you like to check?"
        })

    try:
        amount_match = re.search(r'\$?(\d+(?:\.\d{1,2})?)', user_input)
        category = next((cat for cat in EXPENSE_COLUMNS if cat.lower() in user_input), None)

        if not amount_match or not category:
            return jsonify({
                "response": get_session_specific_error_message(SESSION_TYPE_SUGGESTION)
            })

        suggested_amount = float(amount_match.group(1))
        predicted_total = predict_next_month(df)
        if predicted_total is None:
            end_session()
            return jsonify({"response": "❌ I need at least 3 months of data to make suggestions. Please add more expense records first."})
        
        predicted_breakdown = predict_category_breakdown(df, predicted_total)
        if predicted_breakdown is None:
            end_session()
            return jsonify({"response": "❌ Couldn't calculate category breakdown due to insufficient data."})

        last_income = df.iloc[-1]['Total_Income'] if len(df) > 0 and 'Total_Income' in df.columns else 0
        if last_income <= 0:
            end_session()
            return jsonify({"response": "❌ No valid income data found. Please add income information first."})

        original_savings = calculate_savings_percentage(last_income, predicted_breakdown)

        modified_breakdown = predicted_breakdown.copy()
        modified_breakdown[category] = suggested_amount
        new_savings = calculate_savings_percentage(last_income, modified_breakdown)

        response = []
        if new_savings < original_savings:
            if new_savings < 0:
                response.append("🚨 CRITICAL: This would put you in debt!")
            elif new_savings < 10:
                response.append("⚠️ WARNING: This would severely impact your savings!")
            else:
                response.append("⚠️ CAUTION: This would decrease your savings.")
        else:
            response.append("✅ This looks good for your savings!")

        response.extend([
            f"💰 Financial Impact Analysis:",
            f"Proposed {category} spending: ${suggested_amount:.2f}",
            f"Current predicted {category}: ${predicted_breakdown[category]:.2f}",
            f"Original savings rate: {original_savings:.1f}%",
            f"New savings rate: {new_savings:.1f}%",
            f"Impact: {new_savings - original_savings:+.1f}%"
        ])

        end_session()
        return jsonify({
            "response": "\n".join(response) + "\n\nSay 'menu' for more options."
        })

    except Exception as e:
        print(f"Suggestion session error: {str(e)}")
        end_session()
        return jsonify({"response": "❌ An error occurred while processing your suggestion. Please try again."})

def handle_category_session(df, user_input):
    global EXPENSE_COLUMNS
    if not is_valid_session() or session['session_type'] != SESSION_TYPE_CATEGORY:
        start_session(SESSION_TYPE_CATEGORY)
        return jsonify({
            "response": "🗂 Category Management Session\n"
                       f"Current categories: {', '.join(EXPENSE_COLUMNS)}\n\n"
                       "What would you like to do?\n"
                       "• Type 'add' to add a new category\n"
                       "• Type 'remove' to remove an existing category\n"
                       "• Type 'menu' to return to main menu"
        })

    try:
        # Handle add category
        if user_input.strip() == 'add' or 'add' in user_input:
            session['state'] = 'get_new_category'
            return jsonify({
                "response": "➕ Add New Category\nEnter the name of the new category to add:"
            })
        # Handle remove category  
        elif user_input.strip() == 'remove' or 'remove' in user_input or 'delete' in user_input:
            if len(EXPENSE_COLUMNS) <= 1:
                return jsonify({
                    "response": "❌ Cannot remove categories. You must have at least one expense category."
                })
            session['state'] = 'get_remove_category'
            return jsonify({
                "response": f"➖ Remove Category\nCurrent categories: {', '.join(EXPENSE_COLUMNS)}\nEnter the name of the category to remove:"
            })

        # Handle getting new category name
        if session.get('state') == 'get_new_category':
            new_category = user_input.strip().title()
            if not new_category:
                return jsonify({
                    "response": "❌ Category name cannot be empty. Please enter a valid category name:"
                })
            if new_category in EXPENSE_COLUMNS:
                return jsonify({
                    "response": f"❌ Category '{new_category}' already exists. Please try a different name:"
                })
            if new_category in ['Month', 'Year', 'Total_Income', 'Total_Usage', 'Savings', 'Savings_Percentage']:
                return jsonify({
                    "response": f"❌ '{new_category}' is a reserved name. Please choose a different category name:"
                })
            try:
                df[new_category] = 0
                if save_data(df):
                    get_expense_columns()
                    end_session()
                    return jsonify({
                        "response": f"✅ Successfully added new category: '{new_category}'\n"
                                   f"Updated categories: {', '.join(EXPENSE_COLUMNS)}\n\n"
                                   "From now on, you'll be asked for this category when adding expenses.\n"
                                   "Say 'menu' for more options."
                    })
                else:
                    end_session()
                    return jsonify({"response": "❌ Failed to save the new category. Please try again later."})
            except Exception as e:
                print(f"Error adding category: {str(e)}")
                end_session()
                return jsonify({"response": "❌ Failed to add category due to a system error. Please try a different name."})

        # Handle getting category to remove
        elif session.get('state') == 'get_remove_category':
            category_to_remove = user_input.strip().title()
            if not category_to_remove:
                return jsonify({
                    "response": "❌ Please enter a valid category name to remove:"
                })
            if category_to_remove not in EXPENSE_COLUMNS:
                return jsonify({
                    "response": f"❌ Category '{category_to_remove}' doesn't exist.\n"
                               f"Available categories: {', '.join(EXPENSE_COLUMNS)}\n"
                               "Please try again:"
                })
            session['category_to_remove'] = category_to_remove
            session['state'] = 'confirm_remove'
            return jsonify({
                "response": f"⚠️ Confirm Removal\n"
                           f"Are you sure you want to remove '{category_to_remove}'?\n"
                           f"This will permanently delete all data for this category.\n\n"
                           f"Type 'yes' to confirm or 'no' to cancel:"
            })

        # Handle removal confirmation
        elif session.get('state') == 'confirm_remove':
            if 'yes' in user_input and 'category_to_remove' in session:
                try:
                    category_name = session['category_to_remove']
                    df = df.drop(columns=[category_name])
                    if save_data(df):
                        get_expense_columns()
                        end_session()
                        return jsonify({
                            "response": f"✅ Successfully removed category: '{category_name}'\n"
                                       f"Updated categories: {', '.join(EXPENSE_COLUMNS)}\n\n"
                                       "Say 'menu' for more options."
                        })
                    else:
                        end_session()
                        return jsonify({"response": "❌ Failed to remove category. Please try again later."})
                except Exception as e:
                    print(f"Error removing category: {str(e)}")
                    end_session()
                    return jsonify({"response": "❌ Failed to remove category due to a system error. Please try again."})
            elif 'no' in user_input:
                end_session()
                return jsonify({"response": "Category removal canceled. Say 'menu' for more options."})
            else:
                return jsonify({
                    "response": f"Please type 'yes' to confirm removal of '{session.get('category_to_remove', '')}' or 'no' to cancel:"
                })

        # Handle invalid input in category session
        else:
            return jsonify({
                "response": get_session_specific_error_message(SESSION_TYPE_CATEGORY)
            })

    except Exception as e:
        print(f"Category session error: {str(e)}")
        end_session()
        return jsonify({"response": "❌ An error occurred during category management. Please try again."})

    end_session()
    return jsonify({"response": "Category session ended. Say 'menu' for options."})

def handle_expectation_session(df, user_input):
    if not is_valid_session() or session['session_type'] != SESSION_TYPE_EXPECTATION:
        start_session(SESSION_TYPE_EXPECTATION)
        session['state'] = 'get_income'
        return jsonify({
            "response": "🎯 Financial Expectation Session\n"
                       "Set your financial goals for better predictions:\n\n"
                       "Step 1: Enter your expected monthly income\n"
                       "Step 2: Enter your desired savings percentage\n\n"
                       "Please enter your expected monthly income:"
        })

    try:
        if session['state'] == 'get_income':
            if validate_amount(user_input):
                expected_income = float(user_input)
                if expected_income <= 0:
                    return jsonify({
                        "response": "❌ Income must be greater than 0. Please enter a valid income amount:"
                    })
                session['expected_income'] = expected_income
                session['state'] = 'get_savings_goal'
                return jsonify({
                    "response": f"✅ Expected income set to ${expected_income:.2f}\n\n"
                               f"Now enter your desired savings percentage (e.g., 10 for 10%, 20 for 20%):"
                })
            else:
                return jsonify({
                    "response": "❌ Invalid income amount. Please enter a positive number:"
                })

        elif session['state'] == 'get_savings_goal':
            if validate_amount(user_input):
                savings_pct = float(user_input)
                if 0 <= savings_pct <= 100:
                    session['savings_goal'] = savings_pct
                    session['state'] = 'show_predictions'
                    
                    predicted_total = predict_next_month(df)
                    if predicted_total is None:
                        end_session()
                        return jsonify({"response": "❌ I need at least 3 months of data to make predictions. Please add more expense records first."})
                    
                    # Calculate adjusted prediction based on income and savings goal
                    max_spending = session['expected_income'] * (1 - session['savings_goal']/100)
                    adjusted_prediction = min(predicted_total * 0.9, max_spending)
                    session['adjusted_prediction'] = adjusted_prediction
                    
                    breakdown = predict_category_breakdown(df, adjusted_prediction)
                    if breakdown:
                        session['category_breakdown'] = breakdown
                    
                    actual_savings_amount = session['expected_income'] - adjusted_prediction
                    actual_savings_pct = (actual_savings_amount / session['expected_income']) * 100
                    
                    response = [
                        f"📈 Financial Expectations Analysis",
                        f"Expected Income: ${session['expected_income']:.2f}",
                        f"Savings Goal: {session['savings_goal']:.1f}%",
                        f"Target Savings: ${session['expected_income'] * session['savings_goal']/100:.2f}",
                        f"",
                        f"📊 Adjusted Spending Predictions:",
                        f"Total Expenses: ${adjusted_prediction:.2f}",
                        f"Actual Savings: ${actual_savings_amount:.2f} ({actual_savings_pct:.1f}%)",
                        f"",
                        f"Status: {'✅ Goal Achieved!' if actual_savings_pct >= session['savings_goal'] else '⚠️ Below Target'}",
                        f"",
                        f"Would you like to see the category breakdown? (yes/no)"
                    ]
                    return jsonify({
                        "response": "\n".join(response)
                    })
                else:
                    return jsonify({
                        "response": "❌ Savings percentage must be between 0-100. Please try again:"
                    })
            else:
                return jsonify({
                    "response": "❌ Invalid percentage. Please enter a number between 0-100:"
                })

        elif session['state'] == 'show_predictions':
            if 'yes' in user_input:
                if 'category_breakdown' in session:
                    response = ["💰 Recommended Category Spending:"]
                    total_breakdown = 0
                    for cat, amt in session['category_breakdown'].items():
                        response.append(f"• {cat}: ${amt:.2f}")
                        total_breakdown += amt
                    response.extend([
                        f"",
                        f"Total Categories: ${total_breakdown:.2f}",
                        f"Remaining Budget: ${session['expected_income'] - total_breakdown:.2f}"
                    ])
                    end_session()
                    return jsonify({
                        "response": "\n".join(response) + "\n\nSay 'menu' for more options."
                    })
                else:
                    end_session()
                    return jsonify({"response": "❌ No breakdown available due to insufficient data. Say 'menu' for options."})
            elif 'no' in user_input:
                end_session()
                return jsonify({"response": "Financial expectations set! Say 'menu' for more options."})
            else:
                return jsonify({
                    "response": "Please answer with 'yes' to see category breakdown or 'no' to skip:"
                })

        # Handle invalid input in expectation session
        else:
            return jsonify({
                "response": get_session_specific_error_message(SESSION_TYPE_EXPECTATION)
            })

    except Exception as e:
        print(f"Expectation session error: {str(e)}")
        end_session()
        return jsonify({"response": "❌ An error occurred during expectation setting. Please try again."})

    end_session()
    return jsonify({"response": "Expectation session ended. Say 'menu' for options."})

if __name__ == '__main__':
    app.run(debug=True)
