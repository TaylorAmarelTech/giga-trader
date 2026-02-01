"""Simple syntax checker."""
import ast
import sys

try:
    with open("src/multi_agent_system.py", encoding="utf-8") as f:
        ast.parse(f.read())
    print("Syntax OK")
    sys.exit(0)
except SyntaxError as e:
    print(f"Syntax Error: {e}")
    sys.exit(1)
