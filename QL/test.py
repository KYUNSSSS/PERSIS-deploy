import pytest
import json
import os
import pandas as pd
from QL.appcloud import process_request
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
USER_ID = "3"
ITERATIONS = 1300  # Number of test cycles

# Define specific {persuasive_type, activity} combinations that get "Y"
ALLOWED_COMBINATIONS = {
    ("reminder", "meal planning"),
    ("suggestion", "meal planning"),
    ("reward", "meal planning"),
     ("praise", "meal planning"),
    ("reminder", "water intake"),
    ("suggestion", "water intake"),
    ("reward", "water intake"),
    ("praise", "water intake"),
    # ("reminder", "healthy eating"),
    # ("suggestion", "healthy eating"),
    # ("reward", "healthy eating"),
    # ("praise", "healthy eating"),
    # ("reminder", "portion control"),
    # ("suggestion", "portion control"),
    # ("reward", "portion control"),
    # ("praise", "portion control")
}

# File paths
BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "documents"))

USER_FILE_PATH = os.path.join(BASE_DIR, "userPath", f"{USER_ID}-user.csv")
MESSAGE_FILE_PATH = os.path.join(BASE_DIR, "messagePath", "message.csv")

def setup_files():
    """Setup test files before running tests."""
    os.makedirs("../documents/userPath", exist_ok=True)
    
    if os.path.exists(USER_FILE_PATH):
        os.remove(USER_FILE_PATH)



def test_qlearning_accuracy():
    setup_files()
    yes_count, no_count = 0, 0
    y_true, y_pred = [], []
    for _ in range(ITERATIONS):
        response = process_request({"invoke_type": 2, "userId": USER_ID})

        # Open user file and extract latest row
        file_path = os.path.join(BASE_DIR, "userPath", f"{USER_ID}-user.csv")

        if os.path.exists(file_path):
            user_data = pd.read_csv(file_path)
            latest_row = user_data.iloc[-1]
            print("Latest entry in user file:")
        else:
            print("User file not found.")

        
        print("7888\n")
        question_id = latest_row["id"]
        persuasive_type = latest_row["persuasive_type"].strip()
        activity = latest_row["activity"].strip()

        
        answer = (persuasive_type, activity) in ALLOWED_COMBINATIONS
        y_true.append(1 if answer else 0)

        # # Answering the question
        answer_request = {
            "invoke_type": 3,
            "userId": USER_ID,
            "answer": answer,
            "questionId": question_id
            
        }
        response = process_request(answer_request)
        y_pred.append(1)
        # Output the extracted values
        print(f"Question ID: {question_id}")
        print(f"Persuasive Type: {persuasive_type}")
        print(f"Activity: {activity}")
        print(f"Answer: {answer}")
        if answer:
            yes_count += 1
        else:
            no_count += 1

    accuracy = (yes_count / (yes_count + no_count)) * 100 if (yes_count + no_count) > 0 else 0
    precision = precision_score(y_true, y_pred, zero_division=0) * 100
    recall = recall_score(y_true, y_pred, zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    values = [accuracy, precision, recall, f1]

    plt.figure(figsize=(8, 5))
    plt.bar(metrics, values, color=["blue", "green", "orange", "red"])
    plt.ylim(0, 100)
    plt.ylabel("Percentage")
    plt.title("Q-Learning Model Performance in 1000 Iterations")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    assert accuracy >90  
if __name__ == "__main__":
    pytest.main(["-s","QL\\test.py"])
