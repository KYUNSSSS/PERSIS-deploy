import sys
import json
import pandas as pd
import os
import datetime
import random
import pickle

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
    file_path = os.path.join(os.getcwd(), "documents", "qlearning", f"{user_id}-q_table.pkl")
    global q_table
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            q_table = pickle.load(f)
    

def save_q_table(user_id):
    global q_table
    file_path = os.path.join(os.getcwd(), "documents", "qlearning", f"{user_id}-q_table.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(q_table, f)

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
        # Step 1: Group messages by (persuasive_type, activity) and store max Q-value per group
        message_groups = {}
        for key, value in q_table.items():
            _, persuasive_type, activity = key
            if (persuasive_type, activity) not in message_groups:
                message_groups[(persuasive_type, activity)] = []
            message_groups[(persuasive_type, activity)].append((key, value))

        # Step 2: Sort each group by Q-value (descending)
        for group in message_groups.values():
            group.sort(key=lambda x: x[1], reverse=True)  # Sort messages in each group by Q-value

        # Step 3: Sort groups by their highest Q-value
        sorted_groups = sorted(
            message_groups.items(),
            key=lambda item: max(item[1], key=lambda x: x[1])[1],  # Get the max Q-value from each group
            reverse=True
        )

     
        # Check if there is at least one group with Q-value > 0
        if any(any(q_value > 0 for _, q_value in group) for _, group in sorted_groups):
        # Keep only groups with at least one Q-value > 0
            selected_groups = [
        (key, [(msg, q_value) for msg, q_value in group if q_value > 0])  # Remove 0 and negative Q-values
        for key, group in sorted_groups 
        if any(q_value > 0 for _, q_value in group)  # Keep only groups with at least one Q-value > 0
    ]
            
        else:
            # Keep all groups if all are 0
            selected_groups = sorted_groups
          
        # Step 5: Pick a random message from one of the selected groups
        if selected_groups:
           
            chosen_group = random.choice(selected_groups)[1]  # Get messages from one of the top groups
            message = random.choice(chosen_group)[0]  # Pick a random message from that group
        else:
          
            message = random.choice(list(q_table.keys()))  # Fallback random choice

    epsilon = max(0.01, epsilon * 0.99)  # Gradually decrease exploration rate
    return message

# Update Q-Table based on user response
def update_q_table(message, persuasive_type, activity, reward,question_id, learning_rate=0.001, gamma=0.99):
    key = (message, persuasive_type, activity)
    previous_value = q_table.get(key, 0)
    
    # Reward shaping (increase reward if consistent positive feedback)
    if reward == 1:
        reward += 0.2  # Encourage consistency
        if(question_id==1 or question_id==2):
            q_table[key] = previous_value + learning_rate * (reward + gamma)
        else:
            q_table[key] = previous_value + learning_rate * (reward + gamma * max(q_table.values()) - previous_value)
    else:
        q_table[key]=previous_value-0.5
        if previous_value<0:
            q_table[key]=0

# Load User Data
def load_user_data(user_id):
    file_path = os.path.join(os.getcwd(), "documents", "userPath", f"{user_id}-user.csv")

    # Check if the file exists
    if not os.path.exists(file_path):
        # Create an empty DataFrame with required columns
        empty_df = pd.DataFrame(columns=["id", "message", "persuasive_type", "activity", "yesOrNo", "Date", "Time"])
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save the empty DataFrame as a new CSV file
        empty_df.to_csv(file_path, index=False)
        return empty_df

    # If file exists, load it with proper data types
    return pd.read_csv(file_path, dtype={"id": str, "message": str, "persuasive_type": str, "activity": str, "yesOrNo": str, "Date": str, "Time": str})



# Generate Question (invoke_type == 2)
def generate_question(user_id):
    load_q_table(user_id)
    global q_table
    user_data = load_user_data(user_id)
    messages_df = pd.read_csv("documents\\messagePath\\message.csv")
    
    if messages_df is None or messages_df.empty:
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

    # Generate unique question ID
    question_id = len(user_data) + 1
    if str(question_id) == "1":
        initialize_q_table(messages_df)
        selected_message = random.choice(list(q_table.keys()))
    elif str(question_id) =="2":
        selected_message = random.choice(list(q_table.keys()))
    else:
        selected_message = get_next_message()

    

    # Save question
    new_entry = pd.DataFrame([
        [question_id, selected_message[0], selected_message[1], selected_message[2], "", "", ""]
    ], columns=["id", "message", "persuasive_type", "activity", "yesOrNo", "Date", "Time"])
    
    user_data = pd.concat([user_data, new_entry], ignore_index=True)
    output_dir = os.path.join("..", "documents", "userPath")
    os.makedirs(output_dir, exist_ok=True)  

    file_path = os.path.join(os.getcwd(), "documents", "userPath", f"{user_id}-user.csv")
    user_data.to_csv(file_path, index=False)
    save_q_table(user_id)
    return_json(200, selected_message[0], question_id, selected_message[1], selected_message[2])

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
    file_path = os.path.join(os.getcwd(), "documents", "userPath", f"{user_id}-user.csv")
    # Update answer
    gen_answer = "Y" if answer else "N"
    timestamp = datetime.datetime.now()
    user_data.loc[user_data["id"].astype(str) == str(question_id), ["yesOrNo", "Date", "Time"]] = [gen_answer, str(timestamp.date()), str(timestamp.time())]
    user_data.to_csv(file_path, index=False)

    # Update Q-table
    
    question_data = question_row.iloc[0]
    reward = 1 if answer else 0
    update_q_table(question_data["message"], question_data["persuasive_type"], question_data["activity"], reward,question_id)
    save_q_table(user_id)

    return return_json(200, "Success")

def process_request(request):
    invoke_type = request.get("invoke_type")
    user_id = request.get("username")
    answer = request.get("answer", "")
    question_id = request.get("questionId", None)

    if invoke_type == 2:
        return generate_question(user_id)  # ✅ This now returns JSON
    elif invoke_type == 3:
        return answer_question(user_id, question_id, answer)  # ✅ This now returns JSON
    else:
        return return_json(400, "Invalid invoke type")


# Main Execution
if __name__ == "__main__":
    request = read_input()
    process_request(request)
    