import sys
import json
import pandas as pd
import os
import datetime
import random
import pickle
import io
from supabase import create_client, Client
import base64

# Load Supabase configuration
def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "supabase.json")
    with open(config_path, 'r') as f:
        return json.load(f)

config = load_config()
supabase: Client = create_client(config['supabase']['url'], config['supabase']['key'])

# Load API Input
def read_input():
    input_data = sys.stdin.read()
    return json.loads(input_data)

# Return JSON Response
def return_json(status, message, question_id=None, persuasive_type=None, activity=None):
    response = {
        "status": status,
        "message": message,
        "questionId": question_id,
        "persuasive_type": persuasive_type,
        "activity": activity
    }
    print(json.dumps(response))
    sys.stdout.flush()

# Load Q-Table
q_table = {}
epsilon = 0.5  # Initial exploration rate

def load_q_table(user_id):
    global q_table
    try:
        file_path = f"{config['supabase']['storage']['paths']['qlearning']}/{user_id}-q_table.pkl"
        response = supabase.storage.from_(config['supabase']['storage']['bucket']).download(file_path)
        if response:
            q_table = pickle.loads(response)
    except Exception as e:
        print(f"Error loading Q-table: {str(e)}")
        q_table = {}

def save_q_table(user_id):
    global q_table
    try:
        file_path = f"{config['supabase']['storage']['paths']['qlearning']}/{user_id}-q_table.pkl"
        q_table_bytes = pickle.dumps(q_table)
        supabase.storage.from_(config['supabase']['storage']['bucket']).upload(
            file_path,
            q_table_bytes,
            {"content-type": "application/octet-stream"}
        )
    except Exception as e:
        print(f"Error saving Q-table: {str(e)}")

# Initialize Q-Table
def initialize_q_table(messages_df):
    global q_table
    for _, row in messages_df.iterrows():
        key = (row["message"], row["persuasive_type"], row["activity"])
        if key not in q_table:
            q_table[key] = 0  # Initialize Q-values to zero

# Get the next message using Q-learning
def get_next_message():
    global epsilon
    global q_table
    num = random.random()
    if num < epsilon:  # Exploration: Randomly select a message
        message = random.choice(list(q_table.keys()))
    else:  # Exploitation: Choose from the top Q-value groups
        message_groups = {}
        for key, value in q_table.items():
            _, persuasive_type, activity = key
            if (persuasive_type, activity) not in message_groups:
                message_groups[(persuasive_type, activity)] = []
            message_groups[(persuasive_type, activity)].append((key, value))

        for group in message_groups.values():
            group.sort(key=lambda x: x[1], reverse=True)

        sorted_groups = sorted(
            message_groups.items(),
            key=lambda item: max(item[1], key=lambda x: x[1])[1],
            reverse=True
        )

        if any(any(q_value > 0 for _, q_value in group) for _, group in sorted_groups):
            selected_groups = [
                (key, [(msg, q_value) for msg, q_value in group if q_value > 0])
                for key, group in sorted_groups 
                if any(q_value > 0 for _, q_value in group)
            ]
        else:
            selected_groups = sorted_groups

        if selected_groups:
            chosen_group = random.choice(selected_groups)[1]
            message = random.choice(chosen_group)[0]
        else:
            message = random.choice(list(q_table.keys()))

    epsilon = max(0.01, epsilon * 0.99)
    return message

# Update Q-Table based on user response
def update_q_table(message, persuasive_type, activity, reward, question_id, learning_rate=0.001, gamma=0.99):
    key = (message, persuasive_type, activity)
    previous_value = q_table.get(key, 0)
    
    if reward == 1:
        reward += 0.2
        if question_id == 1 or question_id == 2:
            q_table[key] = previous_value + learning_rate * (reward + gamma)
        else:
            q_table[key] = previous_value + learning_rate * (reward + gamma * max(q_table.values()) - previous_value)
    else:
        q_table[key] = previous_value - 0.5
        if previous_value < 0:
            q_table[key] = 0

# Load User Data
def load_user_data(user_id):
    try:
        file_path = f"{config['supabase']['storage']['paths']['userPath']}/{user_id}-user.csv"
        response = supabase.storage.from_(config['supabase']['storage']['bucket']).download(file_path)
        
        if response:
            return pd.read_csv(io.BytesIO(response), dtype={
                "id": str, "message": str, "persuasive_type": str, 
                "activity": str, "yesOrNo": str, "Date": str, "Time": str
            })
    except Exception as e:
        print(f"Error loading user data: {str(e)}")
    
    # Create empty DataFrame if file doesn't exist
    empty_df = pd.DataFrame(columns=[
        "id", "message", "persuasive_type", "activity", 
        "yesOrNo", "Date", "Time"
    ])
    return empty_df

# Generate Question (invoke_type == 2)
def generate_question(user_id):
    try:
        load_q_table(user_id)
        global q_table
        user_data = load_user_data(user_id)
        
        # Debug print
        print(f"Attempting to load message.csv from: {config['supabase']['storage']['paths']['messagePath']}/message.csv", file=sys.stderr)
        
        try:
            # First check if the file exists
            file_path = f"{config['supabase']['storage']['paths']['messagePath']}/message.csv"
            response = supabase.storage.from_(config['supabase']['storage']['bucket']).download(file_path)
            
            if not response:
                print(f"Error: No response from Supabase for file: {file_path}", file=sys.stderr)
                return return_json(400, "Message file not found in storage")
                
            messages_df = pd.read_csv(io.BytesIO(response))
            print(f"Successfully loaded message.csv with {len(messages_df)} rows", file=sys.stderr)
            
        except Exception as e:
            print(f"Error loading messages: {str(e)}", file=sys.stderr)
            return return_json(400, f"Error loading messages: {str(e)}")

        if messages_df is None or messages_df.empty:
            print("Error: messages_df is empty", file=sys.stderr)
            return return_json(400, "Message database is empty or missing.")

        if not user_data.empty:
            last_row = user_data.iloc[-1]
            last_question_id = last_row["id"]
            last_message = last_row["message"]
            last_type = last_row["persuasive_type"]
            last_activity = last_row["activity"]
            last_answer = last_row["yesOrNo"]

            if pd.isna(last_answer) or last_answer == "":
                return return_json(200, last_message, last_question_id, last_type, last_activity)

        question_id = len(user_data) + 1
        if str(question_id) == "1":
            initialize_q_table(messages_df)
            selected_message = random.choice(list(q_table.keys()))
        elif str(question_id) == "2":
            selected_message = random.choice(list(q_table.keys()))
        else:
            selected_message = get_next_message()

        new_entry = pd.DataFrame([[
            question_id, selected_message[0], selected_message[1], 
            selected_message[2], "", "", ""
        ]], columns=["id", "message", "persuasive_type", "activity", "yesOrNo", "Date", "Time"])
        
        user_data = pd.concat([user_data, new_entry], ignore_index=True)
        
        # Save to Supabase
        try:
            csv_buffer = io.StringIO()
            user_data.to_csv(csv_buffer, index=False)
            supabase.storage.from_(config['supabase']['storage']['bucket']).upload(
                f"{config['supabase']['storage']['paths']['userPath']}/{user_id}-user.csv",
                csv_buffer.getvalue().encode('utf-8'),
                {"content-type": "text/csv"}
            )
            print(f"Successfully saved user data for {user_id}", file=sys.stderr)
        except Exception as e:
            print(f"Error saving user data: {str(e)}", file=sys.stderr)
            return return_json(400, f"Error saving user data: {str(e)}")
        
        save_q_table(user_id)
        return_json(200, selected_message[0], question_id, selected_message[1], selected_message[2])
        
    except Exception as e:
        print(f"Unexpected error in generate_question: {str(e)}", file=sys.stderr)
        return return_json(400, f"Unexpected error: {str(e)}")

# Answer Question (invoke_type == 3)
def answer_question(user_id, question_id, answer):
    user_data = load_user_data(user_id)
    load_q_table(user_id)
    
    if question_id is None:
        return return_json(400, "Question ID is required")

    question_row = user_data[user_data["id"].astype(str) == str(question_id)]
    if question_row.empty:
        return return_json(400, "Failed: Question ID not found.")

    question_answered = question_row.iloc[0]["yesOrNo"]
    if pd.notna(question_answered) and question_answered != "":
        return return_json(400, "Failed: Question ID already answered.")

    # Update answer
    gen_answer = "Y" if answer else "N"
    timestamp = datetime.datetime.now()
    user_data.loc[user_data["id"].astype(str) == str(question_id), ["yesOrNo", "Date", "Time"]] = [
        gen_answer, str(timestamp.date()), str(timestamp.time())
    ]

    # Save to Supabase
    csv_buffer = io.StringIO()
    user_data.to_csv(csv_buffer, index=False)
    supabase.storage.from_(config['supabase']['storage']['bucket']).upload(
        f"{config['supabase']['storage']['paths']['userPath']}/{user_id}-user.csv",
        csv_buffer.getvalue().encode('utf-8'),
        {"content-type": "text/csv"}
    )

    # Update Q-table
    question_data = question_row.iloc[0]
    reward = 1 if answer else 0
    update_q_table(question_data["message"], question_data["persuasive_type"], 
                  question_data["activity"], reward, question_id)
    save_q_table(user_id)

    return return_json(200, "Success")

def process_request(request):
    invoke_type = request.get("invoke_type")
    user_id = request.get("username")
    answer = request.get("answer", "")
    question_id = request.get("questionId", None)

    if invoke_type == 2:
        return generate_question(user_id)
    elif invoke_type == 3:
        return answer_question(user_id, question_id, answer)
    else:
        return return_json(400, "Invalid invoke type")

# Main Execution
if __name__ == "__main__":
    request = read_input()
    process_request(request) 