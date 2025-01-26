from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import re
import os
from datetime import timedelta

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'default_secret_key')  # Use an environment variable for security
app.permanent_session_lifetime = timedelta(minutes=30)  # Configure session lifetime

# Global months list
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

# Expense columns
expense_columns = ['Rent', 'Utilities', 'Groceries', 'Transportation', 'Entertainment', 'Healthcare', 'Misc']

# Load dataset once at the start to avoid reloading on each request
salary_data = None

# Load dataset and process it
def load_salary_data(file_path):
    global salary_data
    if salary_data is None:
        df = pd.read_excel(file_path)
        salary_data = calculate_total_usage(df)
    return salary_data

# Function to calculate 'Total_Usage' and 'Savings_Percentage'
def calculate_total_usage(df):
    if 'Total_Usage' not in df.columns:
        df['Total_Usage'] = df[expense_columns].sum(axis=1)

    # Calculate 'Savings_Percentage' if not present
    if 'Savings_Percentage' not in df.columns and 'Savings' in df.columns:
        df['Savings_Percentage'] = (df['Savings'] / df['Total_Usage']) * 100

    return df

# Predict usage and suggest savings
def predict_usage(df):
    if df.shape[0] < 3:  # Ensure enough data for predictions
        return None

    df['Previous_Month_Usage'] = df['Total_Usage'].shift(1)
    df['2_Months_Ago_Usage'] = df['Total_Usage'].shift(2)
    df.dropna(inplace=True)

    X = df[['Previous_Month_Usage', '2_Months_Ago_Usage']]
    y = df['Total_Usage']

    model = LinearRegression()
    model.fit(X, y)

    recent_usage = df.tail(1)[['Previous_Month_Usage', '2_Months_Ago_Usage']]
    if recent_usage.isnull().values.any():
        return None  # Handle insufficient recent data

    predicted_usage = model.predict(recent_usage)
    return predicted_usage[0]

# Predict expenses by category based on proportions
def predict_expenses_by_category(df, predicted_total_usage):
    total_usage_last_month = df['Total_Usage'].iloc[-1]
    expense_proportions = df[expense_columns].iloc[-1] / total_usage_last_month

    predicted_expenses = expense_proportions * predicted_total_usage
    return predicted_expenses

# Get a summary of expenses for a specific month
def get_expense_summary_by_month(df, month):
    month_data = df[df['Month'] == month]
    if month_data.empty:
        return None
    summary = month_data[expense_columns].iloc[0].to_dict()
    total_usage = month_data['Total_Usage'].iloc[0]
    savings_percentage = month_data['Savings_Percentage'].iloc[0]
    return summary, total_usage, savings_percentage

# Get predictions for all categories with savings
def get_predicted_expenses_with_savings(df):
    predicted_usage = predict_usage(df)
    if predicted_usage is None:
        return None

    predicted_expenses = predict_expenses_by_category(df, predicted_usage)
    predicted_expenses_with_savings = predicted_expenses * 1.1  # Increase by 10% savings
    predicted_expenses_dict = {category: round(amount, 2) for category, amount in zip(expense_columns, predicted_expenses_with_savings)}
    return predicted_expenses_dict

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_chart_data')
def get_chart_data():
    df = load_salary_data('Dataset.xlsx')

    # Ensure the dataset has a 'Month' column
    if 'Month' not in df.columns:
        return jsonify({"error": "Dataset must contain a 'Month' column"})

    # Sort dataframe by the 'Month' column to ensure correct order
    df['Month'] = pd.Categorical(df['Month'], categories=MONTHS, ordered=True)
    df = df.sort_values('Month').reset_index(drop=True)

    predicted_usage = predict_usage(df)
    if predicted_usage is None:
        return jsonify({"error": "Not enough data to make a prediction"})

    # Extract months and usage for the chart
    months = df['Month'].tolist()
    total_usage = df['Total_Usage'].tolist()

    chart_data = {
        'months': months,
        'usage': total_usage
    }

    return jsonify(chart_data)

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    df = load_salary_data('Dataset.xlsx')

    # Retrieve session variables or initialize them
    is_in_prediction_session = session.get('is_in_prediction_session', False)
    is_in_menu_session = session.get('is_in_menu_session', False)  # Track if in menu session

    # Handle welcome message for "Hi", "Hello", or "Hey"
    if any(greeting in user_message.lower() for greeting in ["hi", "hello", "hey"]):
        return jsonify({"response": "Hi! I hope you're doing well. I'm here to assist you regarding your financial advice!"})

    # Handle the 'menu' request to send all the features
    if any(keyword in user_message.lower() for keyword in ["do", "features", "benefits"]):
        # Start the menu session
        session['is_in_menu_session'] = True
        session['feature_index'] = 0  # Reset the feature index
        return jsonify({"response": "You're now in the features menu. Here’s what I can do to help you with your finances!"})

    # Handle next feature in the menu session
    if is_in_menu_session and any(keyword == user_message.lower() for keyword in ["next", "fine", "go ahead"]):
        features = [
            "I can help you in things regarding your financial spends such as:",
            "1. Prediction Session: I can predict your spends for the next and upcoming month which will help to increase your savings.",
            "2. Suggestion Session: I can give you suggestions based on your spending categories like whether you're spending your money in the right way because I respect the value of your income.",
            "3. Update Session: In this session, you can update your next month's spends with me. I'll maintain your spends and you can analyze it whenever you need.",
            "4. Categorizing Session: In this session, you can add or remove any category in your spends. For example, if you're moving to your own house, you can remove the rent category; if you're having children, you can add a category for education.",
            "5. Summary Session: In this session, you can ask me for your summary on any month or category in your expenses and see it visually for better understanding.",
            "These are my features, and feel free to go to any session you need!"
        ]

        feature_index = session.get('feature_index', 0)

        # Check if more features are available
        if feature_index < len(features):
            response = features[feature_index]
            session['feature_index'] = feature_index + 1  # Move to the next feature
            return jsonify({"response": response})

        # All features have been displayed, end menu session
        session['is_in_menu_session'] = False
        session['feature_index'] = 0  # Reset index for future sessions
        return jsonify({"response": "I’ve shown all the features. Feel free to ask if you need more details!"})

    # Handle entering the Prediction Session
    if any(keyword in user_message.lower() for keyword in ["goto prediction", "move on to prediction", "activate prediction"]):
        session['is_in_prediction_session'] = True
        return jsonify({"response": "With your words, let's move on to the prediction session. Prediction session activated. Now you can ask for predictions."})

    # Handle giving predictions
    if any(keyword in user_message.lower() for keyword in ["predicted", "predicted expenses", "how much can i spend"]):
        if not is_in_prediction_session:
            return jsonify({"response": "You must enter the prediction session first. Type 'Activate Prediction session' to begin."})

        predicted_usage = predict_usage(df)
        if predicted_usage is None:
            return jsonify({"response": "Not enough data for prediction."})

        return jsonify({"response": f"Based on your spends in the previous months, your total predicted usage is {round(predicted_usage, 2)}. I can give you predicted expenses for each category if you're interested."})

    # Handle predicted expenses for all categories
    if any(keyword in user_message.lower() for keyword in ["ok then", "all expenses", "individual", "category wise"]):
        if not is_in_prediction_session:
            return jsonify({"response": "You must enter the prediction session first. Type 'Prediction session' to begin."})

        predicted_expenses = get_predicted_expenses_with_savings(df)
        if predicted_expenses is None:
            return jsonify({"response": "I couldn't calculate the predicted expenses."})

        response_text = "\n".join([f"{category}: ${amount}" for category, amount in predicted_expenses.items()])
        response_text += "\nBy using this amount of money in each category, you can increase your savings by 10%."
        return jsonify({"response": response_text})

    # Handle exit from the prediction session
    if any(keyword in user_message.lower() for keyword in ["thanks", "thank you", "let me use it"]):
        session['is_in_prediction_session'] = False
        return jsonify({"response": "You're welcome! I'm always here to help you save your income and spend it the right way."})

    # Check session variables for the update session
    is_in_update_session = session.get('is_in_update_session', False)

    # Update session: "add", "update" or "modify" keywords
    if any(keyword in user_message.lower() for keyword in ["add", "update", "modify"]):
        # Enter Update session and set the dataset to write mode
        session['is_in_update_session'] = True
        session['current_month'] = df['Month'].iloc[-1]  # Current month
        return jsonify({"response": "Sure! Would you like to update your total income too?"})

    # Handle income update (Yes/No)
    if is_in_update_session and any(keyword in user_message.lower() for keyword in ["yes"]):
        return jsonify({"response": "Please enter your total income for the next month."})

    if is_in_update_session and any(keyword in user_message.lower() for keyword in ["no"]):
        return jsonify({"response": "Ok then, we'll update your expenses for the next month by each category. Press ok to continue."})

    # Capture income input and save it
    if is_in_update_session and re.match(r"\d+", user_message):
        total_income = int(user_message)
        # Update total income in the next row of the dataset
        df.loc[df.index[-1] + 1, 'Total_Income'] = total_income
        # Check if income increased or decreased compared to previous month
        previous_income = df['Total_Income'].iloc[-2] if len(df) > 1 else total_income
        income_response = (
            f"Sure! I'm happy to hear that your income increased. Let's update your income too. "
            if total_income > previous_income else
            f"Don't be worried that your income decreased. Let's update your income too. "
        )
        return jsonify({"response": f"{income_response} Ok then, we'll update your expenses for the next month by each category. Press ok to continue."})

    # Expense update flow for each category
    if is_in_update_session and user_message.lower() == "ok":
        session['expense_index'] = 0  # Start with the first expense category
        return jsonify({"response": f"Please enter the amount you would like to spend for {expense_columns[session['expense_index']]} category."})

    # Update expenses based on user input
    if is_in_update_session and re.match(r"\d+", user_message):
        updated_amount = float(user_message)
        category = expense_columns[session['expense_index']]

        # Save the new amount for the category
        df.loc[df.index[-1] + 1, category] = updated_amount

        # Calculate savings percentage and compare with the previous month
        total_usage = df.loc[df.index[-1] + 1, expense_columns].sum()
        savings_percentage = (total_income - total_usage) / total_income * 100
        prev_savings_percentage = df['Savings_Percentage'].iloc[-2] if len(df) > 1 else 0

        savings_response = (
            f"It will increase your savings percentage by {round(savings_percentage - prev_savings_percentage, 2)} compared to the previous month."
            if savings_percentage >= prev_savings_percentage else
            f"Actually, it is not good for you. It will affect your savings percentage by {round(savings_percentage - prev_savings_percentage, 2)} compared to the previous month."
        )

        # Move to the next category
        session['expense_index'] += 1
        if session['expense_index'] < len(expense_columns):
            return jsonify({"response": f"Fine! {savings_response} Ok then, enter the amount for {expense_columns[session['expense_index']]} category."})

        # If it's the last category
        if session['expense_index'] == len(expense_columns):
            # Update the dataset with the new values and exit session
            df.to_excel('Dataset.xlsx', index=False)
            session['is_in_update_session'] = False
            return jsonify({"response": "Ok then, all the expenses are updated. Now you can ask for predictions, suggestions, etc."})

    # Handle exit from the update session (Thanks, Thank you, etc.)
    if is_in_update_session and any(keyword in user_message.lower() for keyword in ["thanks", "thank you", "let me use it"]):
        session['is_in_update_session'] = False
        df.to_excel('Dataset.xlsx', index=False)  # Save the dataset after updates
        return jsonify({"response": "You're welcome! Now you can move on to any other sessions like prediction, suggestion, etc."})

    return jsonify({"response": "I'm sorry, I didn't understand that. Can you please rephrase?"})
if __name__ == '__main__':
    app.run(debug=True)
