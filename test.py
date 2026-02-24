import os

tests = [
    ("data/adult_income.csv", "income"),
    ("data/creditcard.csv", "Class"),
    ("data/loan_default.csv", "Default")
]

for path, label in tests:
    print(f"\nRunning test for {path}")
    os.system(f"python main.py {path} {label}")
